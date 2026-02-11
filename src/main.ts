// Entry point: ORT init, UI bootstrap, game loop

import { getOrt, needsWasmOnly } from './ort'
import { getUI, populateGameSelect, getSelectedModelId, setStatus } from './ui'
import { createRunner, RunnerHandle } from './runner'
import { createInput, InputHandler } from './input'

async function init() {
  // Init ORT
  const ort = await getOrt()
  ort.env.logLevel = 'warning'
  const canUseThreads = typeof SharedArrayBuffer !== 'undefined'
  ort.env.wasm.simd = true
  if (!needsWasmOnly && canUseThreads) {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4
  } else {
    ort.env.wasm.numThreads = 1
  }
  console.log('[Main] ORT initialized', {
    threads: ort.env.wasm.numThreads,
    wasm: needsWasmOnly
  })

  // UI setup
  const ui = getUI()
  populateGameSelect(ui)

  let runner: RunnerHandle | null = null
  let input: InputHandler | null = null
  let running = false
  let animId = 0

  // FPS tracking
  let fpsFrames = 0
  let fpsTime = performance.now()

  async function startGame() {
    // Destroy previous runner (releases ONNX sessions)
    if (runner) {
      running = false
      cancelAnimationFrame(animId)
      input?.destroy()
      input = null
      await runner.destroy()
      runner = null
    }

    ui.startBtn.disabled = true
    ui.gameSelect.disabled = true
    ui.variantSelect.disabled = true

    try {
      const modelId = getSelectedModelId(ui)
      runner = await createRunner({
        canvas: ui.canvas,
        modelId,
        onStatus: (s) => setStatus(ui, s),
      })

      const config = runner.getConfig()
      input = createInput(config, ui.canvas)

      running = true
      fpsFrames = 0
      fpsTime = performance.now()

      function loop() {
        if (!running || !runner || !input) return

        // Feed input action
        runner.inputAction(input.getAction())

        runner.tick().then(() => {
          if (!running) return

          // FPS counter
          fpsFrames++
          const now = performance.now()
          const elapsed = now - fpsTime
          if (elapsed >= 1000) {
            const fps = (fpsFrames / elapsed * 1000).toFixed(1)
            ui.fpsEl.textContent = `${fps} fps`
            fpsFrames = 0
            fpsTime = now
          }

          animId = requestAnimationFrame(loop)
        }).catch(err => {
          console.error('[Main] tick error:', err)
          setStatus(ui, `Error: ${err.message || err}`)
          running = false
        })
      }

      setStatus(ui, 'Playing')
      ui.startBtn.textContent = 'Restart'
      ui.startBtn.disabled = false
      ui.gameSelect.disabled = false
      ui.variantSelect.disabled = false
      animId = requestAnimationFrame(loop)

    } catch (err: any) {
      console.error('[Main] Failed to start:', err)
      setStatus(ui, `Failed: ${err.message || err}`)
      ui.startBtn.disabled = false
      ui.gameSelect.disabled = false
      ui.variantSelect.disabled = false
    }
  }

  ui.startBtn.addEventListener('click', startGame)

  // Prevent context menu on long-press (mobile)
  document.addEventListener('contextmenu', e => e.preventDefault())
}

init().catch(e => {
  console.error('Failed to initialize:', e)
  const status = document.getElementById('status')
  if (status) status.textContent = `Init failed: ${e?.message || e}`
})
