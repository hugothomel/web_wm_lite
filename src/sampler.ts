// Diffusion sampler — ported from web_wm_onnx/src/wm_sampler/diffusionSampler.ts
// Same logic, same buffer reuse pattern

import type * as OrtType from 'onnxruntime-web'

type InputNames = {
  obs: string
  act: string
  hasSigmaCond: boolean
}

type DenoiserConfig = {
  numSteps: number
  sigmaMin: number
  sigmaMax: number
  inputNames?: InputNames
}

export class DiffusionSampler {
  private ort: typeof OrtType
  private numSteps: number
  private sigmaMin: number
  private sigmaMax: number
  private inputNames: InputNames
  private rho = 7

  // Pre-computed sigmas
  private sigmas: Float32Array
  // Reusable buffers
  private xBuf: Float32Array | null = null
  private dBuf: Float32Array | null = null
  private resultBuf: Float32Array | null = null
  private sigmaBuf = new Float32Array(1)
  private sigmaCondBuf = new Float32Array(1)
  private actBuf: BigInt64Array | null = null

  constructor(ort: typeof OrtType, config?: DenoiserConfig) {
    this.ort = ort
    this.numSteps = config?.numSteps ?? 2
    this.sigmaMin = config?.sigmaMin ?? 2e-3
    this.sigmaMax = config?.sigmaMax ?? 5.0
    this.inputNames = config?.inputNames ?? { obs: 'obs', act: 'act', hasSigmaCond: true }

    const num = this.numSteps
    if (num === 1) {
      this.sigmas = new Float32Array([this.sigmaMax, 0])
    } else {
      const minInv = Math.pow(this.sigmaMin, 1 / this.rho)
      const maxInv = Math.pow(this.sigmaMax, 1 / this.rho)
      const l = new Float32Array(num)
      for (let i = 0; i < num; i++) l[i] = i / (num - 1)
      this.sigmas = new Float32Array(num + 1)
      for (let i = 0; i < num; i++) this.sigmas[i] = Math.pow(maxInv + l[i] * (minInv - maxInv), this.rho)
      this.sigmas[num] = 0
    }
  }

  private sampleCount = 0

  async sample(
    denoiser: OrtType.InferenceSession,
    prevObsTC: Float32Array,
    prevActT: Int32Array,
    C: number, H: number, W: number, T: number
  ): Promise<Float32Array> {
    const frameSize = C * H * W

    // Reuse or create buffers
    if (!this.xBuf || this.xBuf.length !== frameSize) {
      this.xBuf = new Float32Array(frameSize)
      this.dBuf = new Float32Array(frameSize)
      this.resultBuf = new Float32Array(frameSize)
    }
    if (!this.actBuf || this.actBuf.length !== T) {
      this.actBuf = new BigInt64Array(T)
    }
    for (let i = 0; i < T; i++) this.actBuf[i] = BigInt(prevActT[i])

    const last = prevObsTC.subarray((T - 1) * frameSize, T * frameSize)
    const x = this.xBuf
    const d = this.dBuf!
    const initialSigma = this.sigmas[0]
    const noiseScale = this.numSteps === 1 ? initialSigma : 0.05
    for (let i = 0; i < frameSize; i++) {
      const u1 = Math.random()
      const u2 = Math.random()
      const gaussian = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2)
      x[i] = last[i] + gaussian * noiseScale
    }

    const sigmas = this.sigmas
    for (let i = 0; i < sigmas.length - 1; i++) {
      const sigma = sigmas[i]
      const nextSigma = sigmas[i + 1]
      const den = await this.denoise(denoiser, x, sigma, prevObsTC, C, H, W, T)
      for (let j = 0; j < frameSize; j++) d[j] = (x[j] - den[j]) / sigma
      const dt = nextSigma - sigma
      for (let j = 0; j < frameSize; j++) x[j] = x[j] + d[j] * dt
    }

    if (this.sampleCount % 60 === 0) {
      console.log(`[Sampler] Frame ${this.sampleCount}`)
    }
    this.sampleCount++

    this.resultBuf!.set(x)
    return this.resultBuf!
  }

  private async denoise(
    denoiser: OrtType.InferenceSession,
    noisyNext: Float32Array,
    sigma: number,
    prevObsTC: Float32Array,
    C: number, H: number, W: number, T: number
  ): Promise<Float32Array> {
    this.sigmaBuf[0] = sigma

    const ort = this.ort
    const feeds: Record<string, OrtType.Tensor> = {
      noisy_next_obs: new ort.Tensor('float32', noisyNext, [1, C, H, W]),
      sigma: new ort.Tensor('float32', this.sigmaBuf, [1]),
      [this.inputNames.obs]: new ort.Tensor('float32', prevObsTC, [1, T * C, H, W]),
      [this.inputNames.act]: new ort.Tensor('int64', this.actBuf!, [1, T])
    }

    if (this.inputNames.hasSigmaCond) {
      this.sigmaCondBuf[0] = this.sigmaMin
      feeds['sigma_cond'] = new ort.Tensor('float32', this.sigmaCondBuf, [1])
    }

    const out = await denoiser.run(feeds)
    const data = out['denoised'].data as Float32Array

    // CRITICAL: Dispose all tensors to free GPU memory
    // Without this, GPU memory leaks ~6 tensors/frame → crash in ~20s
    for (const t of Object.values(feeds)) t.dispose()
    for (const t of Object.values(out)) t.dispose()

    return data
  }
}
