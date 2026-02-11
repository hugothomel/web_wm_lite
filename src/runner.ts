// Runner — simplified orchestrator: env + decoder + renderer
// Client-only, no server mode, no upsamplers

import { WorldModelEnv } from './env'
import { Decoder } from './decoder'
import { Renderer } from './renderer'
import { ModelConfig, getModel, DEFAULT_MODEL } from './models'
import { getUsedBackend } from './ort'

export type RunnerHandle = {
  tick: () => Promise<void>
  inputAction: (action: number) => void
  reset: () => void
  getConfig: () => ModelConfig
  destroy: () => Promise<void>
}

type RunnerOptions = {
  canvas: HTMLCanvasElement
  modelId?: string
  onStatus?: (s: string) => void
}

export async function createRunner(opts: RunnerOptions): Promise<RunnerHandle> {
  const modelId = opts.modelId || DEFAULT_MODEL
  const config = getModel(modelId)

  const renderer = new Renderer(opts.canvas)

  opts.onStatus?.(`Loading ${config.name}...`)
  const env = await WorldModelEnv.create(modelId, opts.onStatus)

  let decoder: Decoder | null = null
  if (config.isLatent && config.paths.decoder && config.decoder) {
    decoder = await Decoder.create(
      config.paths.decoder,
      config.decoder.outputH,
      config.decoder.outputW,
      config.decoder.outputC,
      config.C,
      config.decoder.latentScale,
      opts.onStatus
    )
  }

  const backend = getUsedBackend()
  opts.onStatus?.(`Ready (${backend})`)
  let frameCount = 0

  return {
    async tick() {
      const { obs, c, h, w } = env.currentObs()

      if (config.isLatent && decoder) {
        const rgb = await decoder.decodeAsync(obs, c, h, w)
        const dim = decoder.getOutputDimensions()
        renderer.draw(rgb, dim.c, dim.h, dim.w)
      } else {
        renderer.draw(obs, c, h, w)
      }

      frameCount++
      await env.step()
    },

    inputAction(action: number) {
      env.inputAction(action)
    },

    reset() {
      env.reset()
    },

    getConfig() {
      return config
    },

    // Single active session policy: release ALL sessions
    async destroy() {
      if (decoder) {
        await decoder.destroy()
        decoder = null
      }
      await env.destroy()
    }
  }
}
