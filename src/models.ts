// Model configurations — ported from web_wm_onnx/src/config/models.ts
// Static configs only, no GCS dynamic loading

export interface ModelConfig {
  name: string
  description: string
  C: number
  H: number
  W: number
  T: number
  isLatent?: boolean
  isFP16?: boolean
  numActions: number
  actionNames: string[]
  defaultAction: number
  keymap: Record<string, number>
  upsampler: null
  decoder?: {
    outputH: number
    outputW: number
    outputC: number
    latentScale?: { min: number; max: number }
  } | null
  denoiser: {
    numSteps: number
    sigmaMin: number
    sigmaMax: number
    inputNames?: {
      obs: string
      act: string
      hasSigmaCond: boolean
    }
  }
  paths: {
    denoiser: string
    decoder?: string | null
    initState: string
  }
}

// --- Model Configs ---

const DOOM_DEFEND_LINE: ModelConfig = {
  name: 'Doom Defend Line',
  description: 'VizDoom Defend the Line scenario - shoot enemies!',
  C: 3, H: 48, W: 64, T: 4,
  numActions: 8,
  actionNames: ['NOOP','FORWARD','TURN_LEFT','TURN_RIGHT','STRAFE_LEFT','STRAFE_RIGHT','ATTACK','USE'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 'a':2,'A':2,'ArrowLeft':2, 'd':3,'D':3,'ArrowRight':3, 'q':4,'Q':4, 'e':5,'E':5, ' ':6, 'f':7,'F':7 },
  upsampler: null,
  denoiser: { numSteps: 2, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/doom_defend_line/denoiser.onnx', initState: '/init/doom_defend_line/init_state.json' },
}

const FLAPPY_BIRD: ModelConfig = {
  name: 'Flappy Bird',
  description: 'Classic Flappy Bird - tap to flap!',
  C: 3, H: 64, W: 64, T: 4,
  numActions: 2,
  actionNames: ['DO_NOTHING','FLAP'],
  defaultAction: 0,
  keymap: { ' ':1 },
  upsampler: null,
  denoiser: { numSteps: 2, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'prev_obs', act: 'prev_act', hasSigmaCond: false } },
  paths: { denoiser: '/models/flappy_bird/denoiser.onnx', initState: '/init/flappy_bird/init_state.json' },
}

const ANAMNESIS: ModelConfig = {
  name: 'Anamnesis',
  description: 'Synthetic 3D environment exploration',
  C: 3, H: 64, W: 64, T: 4,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  denoiser: { numSteps: 2, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/anamnesis/denoiser.onnx', initState: '/init/anamnesis/init_state.json' },
}

const ANAMNESIS_LATENT: ModelConfig = {
  name: 'Anamnesis (Latent)',
  description: 'Latent diffusion world model with autoencoder decoder',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3 },
  denoiser: { numSteps: 2, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/anamnesis_latent/denoiser.onnx', decoder: '/models/anamnesis_latent/decoder.onnx', initState: '/init/anamnesis_latent/init_state.json' },
}

const ANAMNESIS_SMALL: ModelConfig = {
  name: 'Anamnesis (Small)',
  description: 'Small model (~20M params)',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3 },
  denoiser: { numSteps: 2, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/anamnesis_small/denoiser.onnx', decoder: '/models/anamnesis_small/decoder.onnx', initState: '/init/anamnesis_small/init_state.json' },
}

const ANAMNESIS_TINY: ModelConfig = {
  name: 'Anamnesis (Tiny)',
  description: 'Smaller model (~10M params)',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3 },
  denoiser: { numSteps: 2, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/anamnesis_tiny/denoiser.onnx', decoder: '/models/anamnesis_tiny/decoder.onnx', initState: '/init/anamnesis_tiny/init_state.json' },
}

const ANAMNESIS_CONSISTENCY: ModelConfig = {
  name: 'Anamnesis (1-Step)',
  description: 'Consistency-distilled 1-step inference - 2x faster!',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3 },
  denoiser: { numSteps: 1, sigmaMin: 0.002, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: false } },
  paths: { denoiser: '/models/anamnesis_consistency/denoiser.onnx', decoder: '/models/anamnesis_consistency/decoder.onnx', initState: '/models/anamnesis_consistency/init_state.json' },
}

const ANAMNESIS_SMALL_DISTILL: ModelConfig = {
  name: 'Anamnesis Small (1-Step)',
  description: 'Consistency-distilled small model',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3 },
  denoiser: { numSteps: 1, sigmaMin: 0.002, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: false } },
  paths: { denoiser: '/models/anamnesis_small_distill/denoiser.onnx', decoder: '/models/anamnesis_small_distill/decoder.onnx', initState: '/init/anamnesis_small_distill/init_state.json' },
}

const MERCURY_FLOW_CONSISTENCY: ModelConfig = {
  name: 'Mercury Flow (1-Step)',
  description: 'Consistency-distilled 1-step mercury flux game',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 2,
  actionNames: ['NOOP','HOLD'],
  defaultAction: 0,
  keymap: { ' ':1 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3, latentScale: { min: -4.397, max: 11.336 } },
  denoiser: { numSteps: 1, sigmaMin: 0.002, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: false } },
  paths: { denoiser: '/models/mercury_flow_consistency/denoiser.onnx', decoder: '/models/mercury_flow_consistency/decoder.onnx', initState: '/models/mercury_flow_consistency/init_state.json' },
}

const MERCURY_FLOW_CONSISTENCY_FP16: ModelConfig = {
  name: 'Mercury Flow FP16 (1-Step)',
  description: 'FP16 weights (~50% smaller)',
  C: 4, H: 64, W: 64, T: 4, isLatent: true, isFP16: true,
  numActions: 2,
  actionNames: ['NOOP','HOLD'],
  defaultAction: 0,
  keymap: { ' ':1 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3, latentScale: { min: -4.397, max: 11.336 } },
  denoiser: { numSteps: 1, sigmaMin: 0.002, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: false } },
  paths: { denoiser: '/models/mercury_flow_consistency_fp16/denoiser.onnx', decoder: '/models/mercury_flow_consistency_fp16/decoder.onnx', initState: '/models/mercury_flow_consistency/init_state.json' },
}

const ANAMNESIS_SMALL_DISTILL_FP16: ModelConfig = {
  name: 'Anamnesis Small FP16 (1-Step)',
  description: 'FP16 weights (~50% smaller)',
  C: 4, H: 64, W: 64, T: 4, isLatent: true, isFP16: true,
  numActions: 7,
  actionNames: ['NOOP','FORWARD','BACKWARD','LEFT_ROTATION','RIGHT_ROTATION','STRAFE_LEFT','STRAFE_RIGHT'],
  defaultAction: 0,
  keymap: { 'w':1,'W':1,'ArrowUp':1, 's':2,'S':2,'ArrowDown':2, 'a':3,'A':3,'ArrowLeft':3, 'd':4,'D':4,'ArrowRight':4, 'q':5,'Q':5, 'e':6,'E':6 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3 },
  denoiser: { numSteps: 1, sigmaMin: 0.002, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: false } },
  paths: { denoiser: '/models/anamnesis_small_distill_fp16/denoiser.onnx', decoder: '/models/anamnesis_small_distill_fp16/decoder.onnx', initState: '/init/anamnesis_small_distill/init_state.json' },
}

const KNIGHTFALL_002: ModelConfig = {
  name: 'Knightfall 002',
  description: 'Turn-based battle RPG - attack, block, magic',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 4,
  actionNames: ['NOOP','ATTACK','BLOCK','MAGIC'],
  defaultAction: 0,
  keymap: { 'a':1,'A':1, 'd':2,'D':2, 's':3,'S':3 },
  upsampler: null,
  decoder: { outputH: 256, outputW: 256, outputC: 3, latentScale: { min: -18.9, max: 15.7 } },
  denoiser: { numSteps: 1, sigmaMin: 0.002, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/knightfall_002/denoiser.onnx', decoder: '/models/knightfall_002/decoder.onnx', initState: '/init/knightfall_002/init_state.json' },
}

const TUBE_RUNNER: ModelConfig = {
  name: 'Tube Runner',
  description: 'Hold to navigate the tube',
  C: 4, H: 64, W: 64, T: 4, isLatent: true,
  numActions: 2,
  actionNames: ['NOOP','HOLD'],
  defaultAction: 0,
  keymap: { ' ':1 },
  upsampler: null,
  decoder: { outputH: 512, outputW: 512, outputC: 3, latentScale: { min: -32.577, max: 26.275 } },
  denoiser: { numSteps: 1, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/tube_runner/denoiser.onnx', decoder: '/models/tube_runner/decoder.onnx', initState: '/init/tube_runner/init_state.json' },
}

const TUBE_RUNNER_FP16: ModelConfig = {
  name: 'Tube Runner FP16',
  description: 'FP16 weights (~50% smaller)',
  C: 4, H: 64, W: 64, T: 4, isLatent: true, isFP16: true,
  numActions: 2,
  actionNames: ['NOOP','HOLD'],
  defaultAction: 0,
  keymap: { ' ':1 },
  upsampler: null,
  decoder: { outputH: 512, outputW: 512, outputC: 3, latentScale: { min: -32.577, max: 26.275 } },
  denoiser: { numSteps: 1, sigmaMin: 2e-3, sigmaMax: 5.0, inputNames: { obs: 'obs', act: 'act', hasSigmaCond: true } },
  paths: { denoiser: '/models/tube_runner_fp16/denoiser.onnx', decoder: '/models/tube_runner_fp16/decoder.onnx', initState: '/init/tube_runner/init_state.json' },
}

// --- Registry ---

export const MODELS: Record<string, ModelConfig> = {
  'doom_defend_line': DOOM_DEFEND_LINE,
  'flappy_bird': FLAPPY_BIRD,
  'anamnesis': ANAMNESIS,
  'anamnesis_latent': ANAMNESIS_LATENT,
  'knightfall_002': KNIGHTFALL_002,
  'anamnesis_small': ANAMNESIS_SMALL,
  'anamnesis_tiny': ANAMNESIS_TINY,
  'anamnesis_consistency': ANAMNESIS_CONSISTENCY,
  'anamnesis_small_distill': ANAMNESIS_SMALL_DISTILL,
  'anamnesis_small_distill_fp16': ANAMNESIS_SMALL_DISTILL_FP16,
  'mercury_flow_consistency': MERCURY_FLOW_CONSISTENCY,
  'mercury_flow_consistency_fp16': MERCURY_FLOW_CONSISTENCY_FP16,
  'tube_runner': TUBE_RUNNER,
  'tube_runner_fp16': TUBE_RUNNER_FP16,
}

export interface GameDefinition {
  id: string
  name: string
  description: string
  variants: { modelId: string; label: string }[]
  defaultVariant: string
}

export const GAMES: GameDefinition[] = [
  {
    id: 'knightfall', name: 'Knightfall',
    description: 'Turn-based battle RPG - attack, block, and magic!',
    variants: [{ modelId: 'knightfall_002', label: 'Standard' }],
    defaultVariant: 'knightfall_002',
  },
  {
    id: 'anamnesis', name: 'Anamnesis',
    description: 'Navigate a synthetic 3D maze',
    variants: [
      { modelId: 'anamnesis_small_distill_fp16', label: 'Small 1-Step FP16 (Recommended)' },
      { modelId: 'anamnesis_small_distill', label: 'Small 1-Step' },
      { modelId: 'anamnesis_consistency', label: '1-Step' },
      { modelId: 'anamnesis_small', label: 'Small 2-Step' },
      { modelId: 'anamnesis_tiny', label: 'Tiny 2-Step' },
    ],
    defaultVariant: 'anamnesis_small_distill_fp16',
  },
  {
    id: 'mercury_flow', name: 'Mercury Flow',
    description: 'Hold to transform reality in this abstract flux world',
    variants: [
      { modelId: 'mercury_flow_consistency_fp16', label: '1-Step FP16 (Recommended)' },
      { modelId: 'mercury_flow_consistency', label: '1-Step' },
      { modelId: 'tube_runner_fp16', label: 'HD 512 FP16' },
      { modelId: 'tube_runner', label: 'HD 512' },
    ],
    defaultVariant: 'mercury_flow_consistency_fp16',
  },
  {
    id: 'doom', name: 'Doom',
    description: 'VizDoom Defend the Line - shoot enemies!',
    variants: [{ modelId: 'doom_defend_line', label: 'Defend Line' }],
    defaultVariant: 'doom_defend_line',
  },
  {
    id: 'flappy', name: 'Flappy Bird',
    description: 'Classic Flappy Bird - tap to flap!',
    variants: [{ modelId: 'flappy_bird', label: 'Classic' }],
    defaultVariant: 'flappy_bird',
  },
]

export const DEFAULT_MODEL = 'anamnesis_small_distill_fp16'

export function getModel(id: string): ModelConfig {
  return MODELS[id] || MODELS[DEFAULT_MODEL]
}
