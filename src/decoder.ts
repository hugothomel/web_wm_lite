// Decoder — pre-allocated buffers, copy-before-dispose, proper session lifecycle

import type * as OrtType from 'onnxruntime-web'
import { getOrt, needsWasmOnly, checkWebGPU } from './ort'

type LatentScale = { min: number; max: number }

export class Decoder {
  private ort: typeof OrtType
  private session: OrtType.InferenceSession
  private latentChannels: number
  private outputH: number
  private outputW: number
  private outputC: number
  private latentScale: LatentScale | null
  // Pre-allocated buffers — zero per-frame allocation
  private rescaleBuf: Float32Array | null = null
  private rgbBuf: Float32Array | null = null
  private decodeCount = 0

  static async create(
    modelPath: string,
    outputH = 256,
    outputW = 256,
    outputC = 3,
    latentChannels = 4,
    latentScale?: LatentScale,
    onStatus?: (s: string) => void
  ): Promise<Decoder> {
    onStatus?.('Loading decoder...')
    const ort = await getOrt()

    let session: OrtType.InferenceSession
    const wasmOpts: OrtType.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
      enableCpuMemArena: true,
      enableMemPattern: true,
    }

    if (needsWasmOnly) {
      session = await ort.InferenceSession.create(modelPath, wasmOpts)
    } else {
      const hasWebGPU = await checkWebGPU()
      if (hasWebGPU) {
        try {
          session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
            executionMode: 'sequential',
          })
        } catch {
          session = await ort.InferenceSession.create(modelPath, wasmOpts)
        }
      } else {
        session = await ort.InferenceSession.create(modelPath, wasmOpts)
      }
    }

    console.log(`[Decoder] inputs: ${session.inputNames.join(', ')}, outputs: ${session.outputNames.join(', ')}`)

    const dec = new Decoder(ort, session, outputH, outputW, outputC, latentChannels, latentScale || null)
    // Pre-allocate RGB output buffer
    dec.rgbBuf = new Float32Array(outputC * outputH * outputW)
    return dec
  }

  private constructor(
    ort: typeof OrtType,
    session: OrtType.InferenceSession,
    outputH: number, outputW: number, outputC: number,
    latentChannels: number,
    latentScale: LatentScale | null
  ) {
    this.ort = ort
    this.session = session
    this.outputH = outputH
    this.outputW = outputW
    this.outputC = outputC
    this.latentChannels = latentChannels
    this.latentScale = latentScale
  }

  async decode(latent: Float32Array, h: number, w: number): Promise<Float32Array> {
    const ort = this.ort

    let latentForDecoder = latent
    if (this.latentScale) {
      const { min, max } = this.latentScale
      if (!this.rescaleBuf || this.rescaleBuf.length !== latent.length) {
        this.rescaleBuf = new Float32Array(latent.length)
      }
      const buf = this.rescaleBuf
      for (let i = 0; i < latent.length; i++) {
        buf[i] = (latent[i] + 1) / 2 * (max - min) + min
      }
      latentForDecoder = buf
    }

    const inputTensor = new ort.Tensor('float32', latentForDecoder, [1, this.latentChannels, h, w])
    const feeds: Record<string, OrtType.Tensor> = { 'latent': inputTensor }

    const out = await this.session.run(feeds)

    // COPY output into pre-allocated buffer BEFORE disposing
    const outData = out['rgb'].data as Float32Array
    if (!this.rgbBuf || this.rgbBuf.length !== outData.length) {
      this.rgbBuf = new Float32Array(outData.length)
    }
    this.rgbBuf.set(outData)

    // Dispose output tensors to free GPU/WASM memory
    for (const t of Object.values(out)) {
      if (t.dispose) t.dispose()
    }
    // Dispose feed tensors to free GPU-side buffer copies
    for (const t of Object.values(feeds)) {
      if (t.dispose) t.dispose()
    }

    return this.rgbBuf
  }

  getOutputDimensions(): { c: number; h: number; w: number } {
    return { c: this.outputC, h: this.outputH, w: this.outputW }
  }

  async decodeAsync(latent: Float32Array, _latentC: number, latentH: number, latentW: number): Promise<Float32Array> {
    const t0 = performance.now()
    const rgb = await this.decode(latent, latentH, latentW)

    if (this.decodeCount % 60 === 0) {
      console.log(`[Decoder] Frame ${this.decodeCount}: ${(performance.now() - t0).toFixed(1)}ms`)
    }
    this.decodeCount++
    return rgb
  }

  async destroy() {
    await this.session.release()
    this.rescaleBuf = null
    this.rgbBuf = null
  }
}
