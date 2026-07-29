import { useCallback, useEffect, useRef, useState } from 'react'
import './Kiosk.css'

const API = '/api'
const COLORS = ['#00e5ff', '#ffd54f', '#ff7043', '#66bb6a', '#ba68c8', '#f06292',
  '#4fc3f7', '#aed581', '#ffb74d', '#9575cd']

// Chế độ Demo: tự động quét lần lượt các khay mẫu có sẵn, không cần bấm tay —
// dùng cho trình chiếu / chạy không người trông.
const SAMPLE_COUNT = 6
const DEMO_DWELL_MS = 7000

const vnd = (n) => `${Number(n || 0).toLocaleString('vi-VN')}đ`

/** Viền mask từng món vẽ bằng SVG polygon theo toạ độ % */
function MaskOverlay({ items, selected, onSelect }) {
  return (
    <svg className="ov-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
      {items.map((it, i) =>
        (it.outlines || []).map((poly, j) =>
          poly.length >= 3 ? (
            <polygon
              key={`${it.group}-${j}`}
              points={poly.map(([x, y]) => `${x},${y}`).join(' ')}
              fill={COLORS[i % COLORS.length]}
              fillOpacity={selected === i ? 0.34 : 0.16}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={selected === i ? 0.7 : 0.4}
              vectorEffect="non-scaling-stroke"
              onClick={() => onSelect(i)}
              style={{ cursor: 'pointer' }}
            />
          ) : null,
        ),
      )}
    </svg>
  )
}

/** Nhãn tên món + định lượng + tiền, đặt ở tâm vùng mask lớn nhất */
function MaskLabels({ items, onSelect }) {
  return items.map((it, i) => {
    const poly = (it.outlines || []).reduce(
      (best, p) => (p.length > (best?.length || 0) ? p : best), null)
    if (!poly || poly.length < 3) return null
    const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length
    const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length
    return (
      <div
        key={it.group}
        className="ov-tag"
        style={{ left: `${cx}%`, top: `${cy}%`, borderColor: COLORS[i % COLORS.length] }}
        onClick={() => onSelect(i)}
      >
        <b>{it.name}</b>
        <span>{it.quantity_display}</span>
        <em>{vnd(it.subtotal)}</em>
      </div>
    )
  })
}

const MODE_LABEL = {
  weight: 'khối lượng', count: 'đếm số lượng',
  piece: 'đếm miếng', volume: 'thể tích', fixed: 'suất cố định',
}

/** Ô chỉnh tay — đơn vị thay đổi theo cách định lượng của món */
function EditRow({ item, onChange }) {
  const spec = {
    weight: ['grams', 'gram', 5, item.grams],
    count: ['count', item.unit || 'cái', 1, item.count],
    piece: ['pieces', 'miếng', 1, item.pieces],
    volume: ['bowls', 'bát', 1, item.bowls],
  }[item.mode]
  if (!spec) return null
  const [field, unit, step, val] = spec
  return (
    <div className="edit-row">
      <span className="edit-lbl">{item.name} — {unit}</span>
      <div className="edit-ctl">
        <button type="button" onClick={() => onChange(field, Math.max(0, (val || 0) - step))}>−</button>
        <input
          type="number" value={val ?? 0} step={step} min={0}
          onChange={(e) => onChange(field, Number(e.target.value))}
        />
        <button type="button" onClick={() => onChange(field, (val || 0) + step)}>+</button>
      </div>
    </div>
  )
}

export default function Kiosk() {
  const [status, setStatus] = useState(null)
  const [imgUrl, setImgUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [selected, setSelected] = useState(null)
  const [showSteps, setShowSteps] = useState(false)
  const [paid, setPaid] = useState(false)
  const [cam, setCam] = useState(false)
  const [txs, setTxs] = useState(null)
  const [demoOn, setDemoOn] = useState(false)
  const [demoIdx, setDemoIdx] = useState(0)

  const fileRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const demoTimerRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/kiosk/status`).then((r) => r.json()).then(setStatus).catch(() => {})
  }, [])

  const loadTxs = useCallback(() => {
    fetch(`${API}/kiosk/transactions?limit=8`).then((r) => r.json()).then(setTxs).catch(() => {})
  }, [])

  const send = useCallback(async (blob) => {
    setBusy(true); setErr(null); setSelected(null); setPaid(false)
    try {
      const fd = new FormData()
      fd.append('file', blob, 'tray.jpg')
      fd.append('canteen_id', 'NA1')
      fd.append('shift', 'Trưa')
      const res = await fetch(`${API}/kiosk/scan`, { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Quét thất bại')
      setResult(data)
      loadTxs()
    } catch (e) {
      setErr(e.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }, [loadTxs])

  const onPick = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    stopDemo()
    setImgUrl(URL.createObjectURL(f))
    send(f)
  }

  /** Ảnh khay mẫu (từ tập val FoodSeg103) để demo không cần chuẩn bị ảnh */
  const useSample = async (n) => {
    const url = `/samples/tray${n}.jpg`
    setImgUrl(url)
    const blob = await (await fetch(url)).blob()
    send(blob)
  }

  const stopDemo = () => {
    clearTimeout(demoTimerRef.current)
    setDemoOn(false)
  }

  const startDemo = () => {
    stopCam()
    setDemoOn(true)
    setDemoIdx(1)
    useSample(1)
  }

  const toggleDemo = () => (demoOn ? stopDemo() : startDemo())

  // Sau khi một khay quét xong (busy chuyển về false), chờ DEMO_DWELL_MS rồi
  // tự động chuyển sang khay mẫu tiếp theo, lặp vô hạn qua SAMPLE_COUNT khay.
  useEffect(() => {
    if (!demoOn || busy) return undefined
    demoTimerRef.current = setTimeout(() => {
      setDemoIdx((i) => {
        const next = (i % SAMPLE_COUNT) + 1
        useSample(next)
        return next
      })
    }, DEMO_DWELL_MS)
    return () => clearTimeout(demoTimerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoOn, busy, result])

  const startCam = async () => {
    stopDemo()
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 1280 },
      })
      streamRef.current = s
      setCam(true)
      requestAnimationFrame(() => { if (videoRef.current) videoRef.current.srcObject = s })
    } catch {
      setErr('Không mở được camera — dùng nút Tải ảnh khay')
    }
  }

  const stopCam = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCam(false)
  }

  /** Bước "Cảm biến phát hiện có khay" — ở demo là nút chụp */
  const capture = () => {
    const v = videoRef.current
    if (!v) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d').drawImage(v, 0, 0)
    const url = c.toDataURL('image/jpeg', 0.9)
    c.toBlob((b) => {
      setImgUrl(url)
      stopCam()
      send(b)
    }, 'image/jpeg', 0.9)
  }

  const editItem = async (idx, field, value) => {
    const items = result.items.map((it, i) => (i === idx ? { ...it, [field]: value } : it))
    const res = await fetch(`${API}/kiosk/recalculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        items: items.map((it) => ({
          group: it.group, grams: it.grams ?? null, count: it.count ?? null,
          pieces: it.pieces ?? null, bowls: it.bowls ?? null,
          area_cm2: it.area_cm2 ?? 0, n_regions: it.n_regions ?? 1,
          confidence: it.confidence ?? 1, outlines: it.outlines ?? [],
        })),
      }),
    })
    const data = await res.json()
    setResult((r) => ({ ...r, items: data.items, total: data.total }))
  }

  const confirmTx = async () => {
    if (result?.transaction_id) {
      await fetch(`${API}/kiosk/confirm`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: result.transaction_id, total_vnd: result.total }),
      })
    }
    setPaid(true); loadTxs()
  }

  const integ = result?.integrity
  const trusted = integ?.trusted
  const modelReady = status?.model?.loaded

  return (
    <div className="kiosk-app">
      <a href="#" className="kiosk-nav-link"
         onClick={(e) => { e.preventDefault(); window.location.hash = '' }}>← Dashboard</a>

      <header className="kiosk-topbar">
        <div className="status">
          <span className="wifi">🍽 Smart-Canteen · Meiko</span>
          <span>
            {demoOn && '🎬 Demo tự động · '}
            {busy ? 'Đang chạy pipeline nhận diện…'
              : err ? `Lỗi: ${err}`
              : result ? result.message
              : modelReady ? 'Sẵn sàng — quét khay' : 'Model chưa nạp'}
          </span>
        </div>
        <div className="total">{result ? vnd(result.total) : '—'}</div>
      </header>

      {status && !modelReady && (
        <div className="kiosk-warn">
          Chưa nạp được weights: {status?.model?.error || 'không rõ'} — chạy
          <code> python -m scripts.train </code> rồi copy best.pt vào
          <code> backend/models/canteen-seg.pt</code>
        </div>
      )}

      <div className="kiosk-body">
        <div className="kiosk-scene">
          {cam ? (
            <video ref={videoRef} autoPlay playsInline muted className="scene-media" />
          ) : imgUrl ? (
            <img src={imgUrl} alt="Khay ăn" className="scene-media" />
          ) : (
            <div className="kiosk-scene-placeholder">
              Camera RGB nhìn từ trên xuống khay ăn
              <small>Chọn khay mẫu, tải ảnh, hoặc bật camera để bắt đầu</small>
            </div>
          )}

          {!cam && result?.items?.length > 0 && (
            <div className="kiosk-overlay">
              <MaskOverlay items={result.items} selected={selected} onSelect={setSelected} />
              <MaskLabels items={result.items} onSelect={setSelected} />
            </div>
          )}

          {busy && (
            <div className="kiosk-scanning">
              <div className="spinner" />
              <span>Phân vùng món → định lượng → tính tiền…</span>
            </div>
          )}

          {demoOn && (
            <div className="demo-badge">
              <span className="demo-dot" /> DEMO · khay {demoIdx}/{SAMPLE_COUNT}
            </div>
          )}

          <div className="scene-controls">
            <div className="samples">
              <span>Khay mẫu</span>
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <button key={n} type="button" className="sample-btn"
                        onClick={() => { stopDemo(); useSample(n) }}
                        disabled={busy || demoOn}>{n}</button>
              ))}
            </div>
            <button type="button"
                    className={`kiosk-btn ${demoOn ? 'kiosk-btn-danger' : 'kiosk-btn-demo'}`}
                    onClick={toggleDemo} disabled={!modelReady}>
              {demoOn ? '⏸ Dừng demo' : '🎬 Demo tự động'}
            </button>
            <input ref={fileRef} type="file" accept="image/*" hidden onChange={onPick} />
            <button type="button" className="kiosk-btn kiosk-btn-primary"
                    onClick={() => fileRef.current?.click()}>📁 Tải ảnh khay</button>
            {cam ? (
              <>
                <button type="button" className="kiosk-btn kiosk-btn-success" onClick={capture}>
                  📸 Chụp &amp; quét
                </button>
                <button type="button" className="kiosk-btn kiosk-btn-secondary" onClick={stopCam}>
                  Tắt camera
                </button>
              </>
            ) : (
              <button type="button" className="kiosk-btn kiosk-btn-secondary" onClick={startCam}>
                🎥 Bật camera
              </button>
            )}
          </div>
        </div>

        <aside className="kiosk-panel">
          {integ && (
            <div className={`verdict ${trusted ? 'ok' : 'warn'}`}>
              <div className="verdict-head">
                {trusted ? '✓ Tin cậy cao — tự động hiển thị'
                         : '! Tin cậy thấp — yêu cầu xác nhận'}
              </div>
              <div className="verdict-meta">
                tin cậy tối thiểu {Math.round((integ.min_confidence || 0) * 100)}%
                {result?.latency_ms != null && ` · ${result.latency_ms}ms`}
              </div>
              {integ.flags?.length > 0 && (
                <ul className="flags">
                  {integ.flags.map((f, i) => (
                    <li key={i} className={f.severity}>{f.message}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className="kiosk-items-list">
            <div className="panel-head">Chi tiết từng món</div>
            {result?.items?.length ? result.items.map((it, i) => (
              <div key={it.group}
                   className={`kiosk-item-row ${selected === i ? 'selected' : ''}`}
                   onClick={() => setSelected(i)}>
                <span className="swatch" style={{ background: COLORS[i % COLORS.length] }} />
                <div className="row-main">
                  <div className="name">
                    {it.name}
                    <small className="mode">{MODE_LABEL[it.mode]}</small>
                    {it.manual && <small className="manual">đã sửa</small>}
                  </div>
                  <div className="meta">{it.formula}</div>
                  <div className="meta dim">
                    {it.area_cm2}cm² · {it.n_regions} vùng · tin cậy {Math.round(it.confidence * 100)}%
                    {it.calib_source === 'nutrition5k' && ' · hệ số đã hiệu chỉnh'}
                  </div>
                </div>
                <div className="sub">{vnd(it.subtotal)}</div>
              </div>
            )) : (
              <div className="empty">{result ? 'Không nhận ra món nào' : 'Chưa có kết quả'}</div>
            )}
            {result && (
              <div className="total-row">
                <span>Tổng {result.items?.length || 0} món</span>
                <span className="amt">{vnd(result.total)}</span>
              </div>
            )}
          </div>

          {result?.items?.length > 0 && (
            <div className="edit-block">
              <div className="panel-head sm">Chỉnh tay định lượng</div>
              {result.items.map((it, i) => (
                <EditRow key={it.group} item={it} onChange={(f, v) => editItem(i, f, v)} />
              ))}
            </div>
          )}

          <div className="kiosk-actions">
            <button type="button" className="kiosk-btn kiosk-btn-secondary"
                    onClick={() => setShowSteps((s) => !s)} disabled={!result}>
              {showSteps ? 'Ẩn' : 'Xem'} pipeline
            </button>
            <button type="button" className="kiosk-btn kiosk-btn-success"
                    disabled={busy || !result?.items?.length} onClick={confirmTx}>
              {trusted ? '💳 Thanh toán' : '✓ Xác nhận & thanh toán'}
            </button>
          </div>
        </aside>
      </div>

      {showSteps && result?.steps && (
        <div className="steps-panel">
          <div className="panel-head">Đường đi của khay qua pipeline</div>
          <ol>
            {result.steps.map((s, i) => (
              <li key={i} className={s.ok ? 'ok' : 'bad'}>
                <b>{s.step}</b>
                <pre>{JSON.stringify(s.detail, null, 1)}</pre>
              </li>
            ))}
          </ol>
        </div>
      )}

      <footer className="kiosk-footer-bar">
        <span>
          {status?.model?.loaded
            ? `${status.model.weights} · ${status.model.classes} class → ` +
              `${status.model.mapped_groups?.length} nhóm món · ${status.model.device}`
            : 'model chưa nạp'}
          {status?.calibration?.eval &&
            ` · hiệu chỉnh gram MAE ${status.calibration.eval.mae_g}g`}
        </span>
        <span>
          {txs?.stats?.transactions
            ? `${txs.stats.transactions} khay · tự động ${Math.round((txs.stats.auto_rate || 0) * 100)}%`
            : ''}
        </span>
      </footer>

      {paid && (
        <div className="kiosk-paid-overlay" onClick={() => setPaid(false)}>
          <div className="kiosk-paid-card" onClick={(e) => e.stopPropagation()}>
            <h2>✓ Đã ghi nhận giao dịch</h2>
            <p>{result?.items?.length} món · mã {result?.transaction_id || '—'}</p>
            <div className="amount">{vnd(result?.total)}</div>
            <button type="button" className="kiosk-btn kiosk-btn-primary"
                    onClick={() => { setPaid(false); setResult(null); setImgUrl(null) }}>
              Khay tiếp theo
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
