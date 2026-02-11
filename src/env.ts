// WorldModelEnv — ported from web_wm_onnx with proper session lifecycle
// KEY FIX: destroy() actually releases ONNX sessions

import type * as OrtType from 'onnxruntime-web'
import { DiffusionSampler } from './sampler'
import { ModelConfig, getModel, DEFAULT_MODEL } from './models'
import { getOrt, needsWasmOnly, checkWebGPU, hasShaderF16, setUsedBackend, getUsedBackend } from './ort'

type ObsState = { obs: Float32Array; c: number; h: number; w: number }

export class WorldModelEnv {
  private sampler: DiffusionSampler
  private denoiser: OrtType.InferenceSession
  private obsBuf: Float32Array
  private actBuf: Int32Array
  private currentAction = 0
  private config: ModelConfig
  private C: number
  private H: number
  private W: number
  private T: number
  private numActions: number

  static async create(
    modelId: string = DEFAULT_MODEL,
    onStatus?: (s: string) => void
  ): Promise<WorldModelEnv> {
    const config = getModel(modelId)
    onStatus?.(`Loading ${config.name}...`)

    const shaderF16Available = await hasShaderF16()
    const ort = await getOrt()

    let denoiser: OrtType.InferenceSession

    const webgpuOpts: OrtType.InferenceSession.SessionOptions = {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    }
    const wasmOpts: OrtType.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
      enableCpuMemArena: true,
      enableMemPattern: true,
    }

    const fp16NeedsWasm = config.isFP16 && !shaderF16Available
    console.log(`[Env] isFP16=${config.isFP16}, shaderF16=${shaderF16Available}, fp16NeedsWasm=${fp16NeedsWasm}, needsWasmOnly=${needsWasmOnly}`)

    if (needsWasmOnly || fp16NeedsWasm) {
      const reason = needsWasmOnly ? 'WASM-only mode' : 'FP16 model without shader-f16'
      console.log(`[Env] Using WASM backend: ${reason}`)
      onStatus?.(`Loading model (WASM: ${reason})...`)
      denoiser = await ort.InferenceSession.create(config.paths.denoiser, wasmOpts)
      setUsedBackend('wasm')
    } else {
      const hasWebGPU = await checkWebGPU()
      console.log(`[Env] WebGPU available: ${hasWebGPU}`)
      if (hasWebGPU) {
        try {
          onStatus?.('Loading model (WebGPU)...')
          denoiser = await ort.InferenceSession.create(config.paths.denoiser, webgpuOpts)
          setUsedBackend('webgpu')
          console.log('[Env] WebGPU session created')
        } catch (e) {
          console.warn('[Env] WebGPU failed, falling back to WASM:', e)
          onStatus?.('Loading model (WASM fallback)...')
          denoiser = await ort.InferenceSession.create(config.paths.denoiser, wasmOpts)
          setUsedBackend('wasm')
        }
      } else {
        console.log('[Env] No WebGPU, using WASM')
        onStatus?.('Loading model (WASM)...')
        denoiser = await ort.InferenceSession.create(config.paths.denoiser, wasmOpts)
        setUsedBackend('wasm')
      }
    }

    console.log(`[Env] Backend: ${getUsedBackend()}`)
    console.log(`[Env] Session inputs: ${denoiser.inputNames.join(', ')}`)
    console.log(`[Env] Session outputs: ${denoiser.outputNames.join(', ')}`)

    const sampler = new DiffusionSampler(ort, {
      ...config.denoiser,
      isFP16: config.isFP16 ?? false,
    } as any)
    const env = new WorldModelEnv(sampler, denoiser, config)
    await env.reset()
    return env
  }

  private constructor(sampler: DiffusionSampler, denoiser: OrtType.InferenceSession, config: ModelConfig) {
    this.sampler = sampler
    this.denoiser = denoiser
    this.config = config
    this.C = config.C
    this.H = config.H
    this.W = config.W
    this.T = config.T
    this.numActions = config.numActions

    const obsLen = this.T * this.C * this.H * this.W
    const actLen = this.T
    if (typeof SharedArrayBuffer !== 'undefined') {
      this.obsBuf = new Float32Array(new SharedArrayBuffer(obsLen * 4))
      this.actBuf = new Int32Array(new SharedArrayBuffer(actLen * 4))
    } else {
      this.obsBuf = new Float32Array(obsLen)
      this.actBuf = new Int32Array(actLen)
    }
  }

  getConfig(): ModelConfig { return this.config }

  async reset() {
    try {
      const res = await fetch(this.config.paths.initState)
      if (res.ok) {
        const j = await res.json()
        const T = (j.T ?? this.T) | 0
        const C = (j.C ?? this.C) | 0
        const H = (j.H ?? this.H) | 0
        const W = (j.W ?? this.W) | 0
        if (T === this.T && C === this.C && H === this.H && W === this.W) {
          const ob = new Float32Array(j.obs_buffer)
          const ab = new Int32Array(j.act_buffer)
          if (ob.length === this.obsBuf.length) this.obsBuf.set(ob)
          if (ab.length === this.actBuf.length) this.actBuf.set(ab)
          console.log('[Env] Init state loaded')
          return
        }
      }
    } catch (e) {
      console.warn('[Env] Init state error:', e)
    }
    this.obsBuf.fill(0)
    this.actBuf.fill(0)
  }

  currentObs(): ObsState {
    const offset = (this.T - 1) * this.C * this.H * this.W
    return { obs: this.obsBuf.subarray(offset, offset + this.C * this.H * this.W), c: this.C, h: this.H, w: this.W }
  }

  inputAction(action: number) {
    this.currentAction = Math.max(0, Math.min(this.numActions - 1, action))
  }

  private rollBuffers(nextObs: Float32Array, action: number) {
    const frameSize = this.C * this.H * this.W
    this.obsBuf.copyWithin(0, frameSize)
    this.obsBuf.set(nextObs, (this.T - 1) * frameSize)
    this.actBuf.copyWithin(0, 1)
    this.actBuf[this.T - 1] = action
  }

  async step() {
    this.actBuf[this.T - 1] = this.currentAction

    // Pass buffers directly — sampler only reads them, and rollBuffers runs after sample completes
    const next = await this.sampler.sample(this.denoiser, this.obsBuf, this.actBuf, this.C, this.H, this.W, this.T)
    this.rollBuffers(next, this.currentAction)
  }

  // KEY FIX: Actually release the ONNX session
  async destroy() {
    await this.denoiser.release()
  }
}
