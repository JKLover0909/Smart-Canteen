import { useCallback, useEffect, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const API = '/api'
const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#a855f7', '#ef4444']
const WEEKDAYS = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']

function formatDate(iso) {
  const d = new Date(iso)
  return d.toLocaleDateString('vi-VN', { weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric' })
}

function tomorrowISO() {
  const d = new Date()
  d.setDate(d.getDate() + 1)
  return d.toISOString().slice(0, 10)
}

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [dashboard, setDashboard] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [targetDate, setTargetDate] = useState(tomorrowISO())
  const [weather, setWeather] = useState('nắng')
  const [activeShift, setActiveShift] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadDashboard = useCallback(async () => {
    const res = await fetch(`${API}/dashboard`)
    if (!res.ok) throw new Error('Không tải được dashboard')
    return res.json()
  }, [])

  const loadForecast = useCallback(async (date, w) => {
    const params = new URLSearchParams({ target_date: date, weather: w })
    const res = await fetch(`${API}/forecast?${params}`)
    if (!res.ok) throw new Error('Không tải được dự báo')
    return res.json()
  }, [])

  useEffect(() => {
    let cancelled = false
    async function init() {
      setLoading(true)
      setError(null)
      try {
        const [dash, fc] = await Promise.all([
          loadDashboard(),
          loadForecast(targetDate, weather),
        ])
        if (!cancelled) {
          setDashboard(dash)
          setForecast(fc)
        }
      } catch (e) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    init()
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const refreshForecast = async () => {
    setLoading(true)
    setError(null)
    try {
      const fc = await loadForecast(targetDate, weather)
      setForecast(fc)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading && !dashboard) {
    return <div className="loading">Đang tải Smart-Canteen demo…</div>
  }

  if (error && !dashboard) {
    return (
      <div className="loading">
        <p>Lỗi: {error}</p>
        <p style={{ marginTop: '1rem', fontSize: '0.85rem' }}>
          Hãy chạy backend: <code>cd backend && pip install -r requirements.txt && python run.py</code>
        </p>
      </div>
    )
  }

  const trendData = (dashboard?.daily_trend || []).map((d) => ({
    ...d,
    label: d.date.slice(5),
  }))

  const currentShift = forecast?.shifts?.[activeShift]

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">🍽</div>
          <div>
            <h1>Smart-Canteen</h1>
            <span>FoodsAI Demo v0.1</span>
          </div>
        </div>
        <nav className="nav">
          {[
            ['dashboard', '📊 Tổng quan'],
            ['forecast', '🔮 Dự báo định lượng'],
            ['ingredients', '🥬 Nguyên liệu'],
          ].map(([id, label]) => (
            <button
              key={id}
              className={`nav-item ${tab === id ? 'active' : ''}`}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
          <a href="#meiko" className="nav-item" style={{ textDecoration: 'none', display: 'block' }}>
            🖥️ Meiko cân tính tiền
          </a>
        </nav>
      </aside>

      <main className="main">
        <header className="header">
          <div>
            <h2>
              {tab === 'dashboard' && 'Tổng quan nhà ăn'}
              {tab === 'forecast' && 'Dự báo suất ăn ngày mai'}
              {tab === 'ingredients' && 'Định lượng nguyên liệu'}
            </h2>
            <p>Nhà ăn công nghiệp · 3 ca · 8 món · ML Gradient Boosting</p>
          </div>
          <span className="badge">Demo live</span>
        </header>

        {tab === 'dashboard' && dashboard && (
          <>
            <div className="stats-grid">
              <StatCard label="Suất ăn 7 ngày qua" value={dashboard.total_servings_7d.toLocaleString('vi-VN')} sub={`~${dashboard.avg_daily_servings}/ngày`} accent />
              <StatCard label="Độ chính xác dự báo" value={`${dashboard.forecast_accuracy_pct}%`} sub="MAPE 8.8%" blue />
              <StatCard label="Tỷ lệ lãng phí" value={`${dashboard.waste_rate_pct}%`} sub="↓ so với tuần trước" warn />
              <StatCard label="Đăng ký / Thực tế hôm nay" value={`${dashboard.registered_today} / ${dashboard.checked_in_today}`} sub="Chốt 2h trước ca ăn" />
            </div>

            <div className="grid-2">
              <div className="panel">
                <h3>Xu hướng suất ăn (7 ngày)</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={trendData}>
                    <defs>
                      <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#22c55e" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#2d3f56" strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke="#8b9cb3" fontSize={11} />
                    <YAxis stroke="#8b9cb3" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#1a2332', border: '1px solid #2d3f56' }} />
                    <Area type="monotone" dataKey="total" stroke="#22c55e" fill="url(#g)" name="Suất ăn" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="panel">
                <h3>Phân bổ theo ca</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie
                      data={dashboard.shift_split}
                      dataKey="total"
                      nameKey="shift_name"
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={3}
                    >
                      {dashboard.shift_split.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#1a2332', border: '1px solid #2d3f56' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel">
              <h3>Top món ăn tuần này</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={dashboard.top_dishes} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid stroke="#2d3f56" strokeDasharray="3 3" />
                  <XAxis type="number" stroke="#8b9cb3" fontSize={11} />
                  <YAxis type="category" dataKey="dish_name" stroke="#8b9cb3" fontSize={11} width={75} />
                  <Tooltip contentStyle={{ background: '#1a2332', border: '1px solid #2d3f56' }} />
                  <Bar dataKey="total" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Suất" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {tab === 'forecast' && forecast && (
          <>
            <div className="controls">
              <input type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
              <select value={weather} onChange={(e) => setWeather(e.target.value)}>
                <option value="nắng">☀️ Nắng</option>
                <option value="âm u">⛅ Âm u</option>
                <option value="mưa">🌧 Mưa</option>
              </select>
              <button onClick={refreshForecast}>Cập nhật dự báo</button>
            </div>

            <div className="stats-grid">
              <StatCard label="Ngày dự báo" value={formatDate(forecast.target_date)} sub={`Thứ ${WEEKDAYS[forecast.weekday] || '?'}`} />
              <StatCard label="Tổng suất dự kiến" value={forecast.total_servings.toLocaleString('vi-VN')} sub={`Thời tiết: ${forecast.weather}`} accent />
              <StatCard label="Ca trưa (peak)" value={forecast.shifts?.[1]?.total_predicted?.toLocaleString('vi-VN') || '—'} sub="11:00 – 13:00" blue />
            </div>

            <div className="shift-tabs">
              {forecast.shifts?.map((s, i) => (
                <button
                  key={s.shift}
                  className={`shift-tab ${activeShift === i ? 'active' : ''}`}
                  onClick={() => setActiveShift(i)}
                >
                  {s.shift_name} ({s.total_predicted})
                </button>
              ))}
            </div>

            <div className="panel">
              <h3>{currentShift?.shift_name} · {currentShift?.time}</h3>
              <table>
                <thead>
                  <tr>
                    <th>Món</th>
                    <th>Loại</th>
                    <th>Dự báo</th>
                    <th>Khuyến nghị nấu</th>
                    <th>Độ tin cậy</th>
                  </tr>
                </thead>
                <tbody>
                  {currentShift?.dishes?.map((d) => (
                    <tr key={d.dish_id}>
                      <td><strong>{d.dish_name}</strong></td>
                      <td><span className="tag">{d.category}</span></td>
                      <td>{d.predicted_servings}</td>
                      <td>{d.recommended_prep}</td>
                      <td className="confidence">{d.confidence_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {tab === 'ingredients' && forecast && (
          <>
            <div className="controls">
              <input type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
              <select value={weather} onChange={(e) => setWeather(e.target.value)}>
                <option value="nắng">☀️ Nắng</option>
                <option value="âm u">⛅ Âm u</option>
                <option value="mưa">🌧 Mưa</option>
              </select>
              <button onClick={refreshForecast}>Cập nhật</button>
            </div>

            <div className="panel">
              <h3>Nguyên liệu cần chuẩn bị · {formatDate(forecast.target_date)}</h3>
              <p style={{ color: 'var(--muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Tính từ dự báo {forecast.total_servings} suất + buffer 5%
              </p>
              <div className="ingredient-list">
                {forecast.ingredients?.map((ing) => (
                  <div key={ing.name} className="ingredient-item">
                    <div className="name">{ing.name}</div>
                    <div className="qty">{ing.quantity} {ing.unit}</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        <p className="footer-note">
          Smart-Canteen Demo · Dữ liệu mô phỏng 90 ngày · Gradient Boosting Regressor · Không phải production
        </p>
      </main>
    </div>
  )
}

function StatCard({ label, value, sub, accent, blue, warn }) {
  const cls = ['stat-card', accent && 'accent', blue && 'blue', warn && 'warn'].filter(Boolean).join(' ')
  return (
    <div className={cls}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}
