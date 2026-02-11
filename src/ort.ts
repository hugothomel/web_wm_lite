// ORT loading, WebGPU detection, backend selection
// Simplified from web_wm_onnx: no blocklist, no benchmark, no backend choice UI

export type BackendChoice = 'auto' | 'webgpu' | 'wasm'

function shouldUseWasmOnly(): boolean {
  const choice = (localStorage.getItem('ort-backend') as BackendChoice) || 'auto'
  if (choice === 'wasm') return true
  if (choice === 'webgpu') return false
  return false
}

export const needsWasmOnly = shouldUseWasmOnly()

let ortModule: typeof import('onnxruntime-web') | null = null

export async function getOrt(): Promise<typeof import('onnxruntime-web')> {
  if (ortModule) return ortModule

  if (needsWasmOnly) {
    console.log('[ORT] Loading WASM-only module...')
    ortModule = await import('onnxruntime-web/wasm')
    ortModule.env.wasm.wasmPaths = '/wasm/'
    ortModule.env.wasm.numThreads = navigator.hardwareConcurrency || 4
    ortModule.env.wasm.proxy = false
    ortModule.env.wasm.simd = true
  } else {
    console.log('[ORT] Loading full ORT module with WebGPU...')
    ortModule = await import('onnxruntime-web')
    ortModule.env.wasm.wasmPaths = '/wasm/'
  }

  return ortModule
}

export async function checkWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as any).gpu
    if (!gpu) return false
    const adapter = await gpu.requestAdapter()
    if (!adapter) return false
    const info = await adapter.requestAdapterInfo?.() || {}
    console.log(`[ORT] WebGPU adapter: ${info.vendor || 'unknown'} - ${info.device || 'unknown'}`)
    return true
  } catch {
    return false
  }
}

let shaderF16Support: boolean | null = null

export async function hasShaderF16(): Promise<boolean> {
  if (shaderF16Support !== null) return shaderF16Support
  try {
    const gpu = (navigator as any).gpu
    if (!gpu) { shaderF16Support = false; return false }
    const adapter = await gpu.requestAdapter()
    if (!adapter) { shaderF16Support = false; return false }
    shaderF16Support = adapter.features.has('shader-f16') as boolean
    console.log(`[ORT] shader-f16 support: ${shaderF16Support}`)
    return shaderF16Support!
  } catch {
    shaderF16Support = false
    return false
  }
}

// Track which backend we're actually using
let usedBackend: 'webgpu' | 'wasm' | 'unknown' = 'unknown'
export function setUsedBackend(b: 'webgpu' | 'wasm') { usedBackend = b }
export function getUsedBackend(): string { return usedBackend }
