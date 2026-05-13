import { useState, useEffect, useCallback } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Lock, Search, Database, Server, ChevronDown, ChevronRight,
  RefreshCw, Smartphone, Activity, Layers, Trash2, RotateCcw,
  AlertTriangle, Power,
} from 'lucide-react'
import api, {
  adminDeleteSession, adminRestoreSession, adminPurgeDeleted,
  adminDeleteLog, adminTogglePhoneModel,
} from '@/services/api'

interface DebugInfo {
  health: Record<string, unknown>
  config: Record<string, string>
  tables: Record<string, number>
}

interface AdminSession {
  id: number
  session_id: string
  mode: string
  total_frames: number
  final_model_name: string | null
  final_confidence: number | null
  total_processing_ms: number
  is_deleted: boolean
  created_at: string | null
  deleted_at: string | null
}

interface AdminLog {
  id: number
  session_id: string
  frame_index: number
  filename: string
  detection_count: number
  detection_confidence: number | null
  classification_model_name: string | null
  classification_confidence: number | null
  quality_score: number | null
  is_best_frame: boolean
  processing_time_ms: number
  created_at: string | null
}

interface PhoneModelItem {
  id: number
  model_key: string
  manufacturer: string | null
  brand: string | null
  model_name: string | null
  is_active: boolean
}

type Tab = 'overview' | 'sessions' | 'logs' | 'phones'

function formatModelName(name: string | null) {
  if (!name) return '—'
  const labels: Record<string, string> = {
    HUAWEI_NOVA_10: 'HUAWEI nova 10',
    REDMI_K80_Pro: 'Redmi K80 Pro',
    iPhone_13: 'iPhone 13',
  }
  return labels[name] || name.replace(/_/g, ' ')
}

export default function AdminPage() {
  const [authed, setAuthed] = useState(false)
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const handleLogin = async () => {
    try {
      await api.get('/admin/debug', { params: { key: password } })
      setAuthed(true)
      setError('')
    } catch {
      setError('密码错误')
    }
  }

  if (!authed) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-full max-w-sm p-8 rounded-2xl border border-border bg-card shadow-lg">
          <div className="flex flex-col items-center mb-6">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-3">
              <Lock className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-xl font-bold">管理员后台</h1>
            <p className="text-sm text-muted-foreground mt-1">请输入管理密码</p>
          </div>
          <form
            onSubmit={(e) => { e.preventDefault(); void handleLogin() }}
            className="space-y-4"
          >
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="输入密码..."
              autoFocus
              className="w-full rounded-lg border border-border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            {error && <p className="text-sm text-destructive">{error}</p>}
            <Button type="submit" className="w-full">
              登录
            </Button>
          </form>
        </div>
      </div>
    )
  }

  return <AdminDashboard adminKey={password} />
}

function AdminDashboard({ adminKey }: { adminKey: string }) {
  const [tab, setTab] = useState<Tab>('overview')
  const [debug, setDebug] = useState<DebugInfo | null>(null)
  const [sessions, setSessions] = useState<AdminSession[]>([])
  const [sessionsTotal, setSessionsTotal] = useState(0)
  const [logs, setLogs] = useState<AdminLog[]>([])
  const [logsTotal, setLogsTotal] = useState(0)
  const [phones, setPhones] = useState<PhoneModelItem[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [expandedSession, setExpandedSession] = useState<string | null>(null)
  const [sessionLogs, setSessionLogs] = useState<AdminLog[]>([])
  const [includeDeleted, setIncludeDeleted] = useState(false)
  const [sessionsPage, setSessionsPage] = useState(1)
  const [logsPage, setLogsPage] = useState(1)

  const loadDebug = useCallback(async () => {
    try {
      const { data } = await api.get('/admin/debug', { params: { key: adminKey } })
      setDebug(data)
    } catch { /* ignore */ }
  }, [adminKey])

  const loadSessions = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await api.get('/admin/sessions', {
        params: { key: adminKey, search, include_deleted: includeDeleted, page: sessionsPage, size: 30 },
      })
      setSessions(data.items)
      setSessionsTotal(data.total)
    } catch { /* ignore */ }
    setLoading(false)
  }, [adminKey, search, includeDeleted, sessionsPage])

  const loadLogs = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await api.get('/admin/logs', {
        params: { key: adminKey, search, page: logsPage, size: 50 },
      })
      setLogs(data.items)
      setLogsTotal(data.total)
    } catch { /* ignore */ }
    setLoading(false)
  }, [adminKey, search, logsPage])

  const loadPhones = useCallback(async () => {
    try {
      const { data } = await api.get('/admin/phone-models', { params: { key: adminKey } })
      setPhones(data)
    } catch { /* ignore */ }
  }, [adminKey])

  const loadSessionLogs = async (sessionId: string) => {
    if (expandedSession === sessionId) {
      setExpandedSession(null)
      return
    }
    try {
      const { data } = await api.get('/admin/logs', {
        params: { key: adminKey, session_id: sessionId, size: 100 },
      })
      setSessionLogs(data.items)
      setExpandedSession(sessionId)
    } catch { /* ignore */ }
  }

  // ---------- 数据库写操作 ----------

  const handleDeleteSession = async (sessionId: string) => {
    if (!confirm(`确定永久删除会话 ${sessionId} 及其全部日志？此操作不可撤销`)) return
    try {
      await adminDeleteSession(sessionId, adminKey)
      toast.success('会话已物理删除')
      loadSessions(); loadDebug()
    } catch (e: any) {
      toast.error('删除失败', { description: e?.message })
    }
  }

  const handleRestoreSession = async (sessionId: string) => {
    try {
      await adminRestoreSession(sessionId, adminKey)
      toast.success('会话已恢复')
      loadSessions(); loadDebug()
    } catch (e: any) {
      toast.error('恢复失败', { description: e?.message })
    }
  }

  const handlePurgeDeleted = async () => {
    if (!confirm('确定永久清除所有已软删除的会话及其日志？此操作不可撤销')) return
    try {
      const res = await adminPurgeDeleted(adminKey)
      toast.success(`已清除 ${res.purged_sessions} 个会话 / ${res.purged_logs} 条日志`)
      loadSessions(); loadDebug()
    } catch (e: any) {
      toast.error('清除失败', { description: e?.message })
    }
  }

  const handleDeleteLog = async (logId: number) => {
    if (!confirm(`确定删除日志 #${logId}？`)) return
    try {
      await adminDeleteLog(logId, adminKey)
      toast.success('日志已删除')
      loadLogs(); loadDebug()
    } catch (e: any) {
      toast.error('删除失败', { description: e?.message })
    }
  }

  const handleTogglePhone = async (modelId: number) => {
    try {
      const res = await adminTogglePhoneModel(modelId, adminKey)
      toast.success(`型号已${res.is_active ? '启用' : '停用'}`)
      loadPhones()
    } catch (e: any) {
      toast.error('切换失败', { description: e?.message })
    }
  }

  useEffect(() => { loadDebug() }, [loadDebug])

  useEffect(() => {
    if (tab === 'sessions') loadSessions()
    else if (tab === 'logs') loadLogs()
    else if (tab === 'phones') loadPhones()
  }, [tab, loadSessions, loadLogs, loadPhones])

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'overview', label: '系统概览', icon: <Activity className="h-4 w-4" /> },
    { key: 'sessions', label: '检测会话', icon: <Layers className="h-4 w-4" /> },
    { key: 'logs', label: '日志明细', icon: <Database className="h-4 w-4" /> },
    { key: 'phones', label: '手机型号库', icon: <Smartphone className="h-4 w-4" /> },
  ]

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-3">
              <Server className="h-5 w-5 text-primary" />
              <span className="font-bold">PhoneScope 管理后台</span>
              <Badge variant="outline" className="text-xs">Admin</Badge>
            </div>
            <Button variant="ghost" size="sm" onClick={loadDebug} className="gap-1">
              <RefreshCw className="h-3.5 w-3.5" /> 刷新
            </Button>
          </div>
          <div className="flex gap-1 -mb-px">
            {tabs.map((t) => (
              <button
                key={t.key}
                onClick={() => { setTab(t.key); setSearch('') }}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                  tab === t.key
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {tab !== 'overview' && tab !== 'phones' && (
          <div className="flex gap-3 mb-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="搜索会话ID、型号、文件名..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    if (tab === 'sessions') loadSessions()
                    else loadLogs()
                  }
                }}
                className="w-full pl-9 pr-4 py-2 rounded-lg border border-border bg-card text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => { tab === 'sessions' ? loadSessions() : loadLogs() }}
              className="gap-1.5"
            >
              <Search className="h-3.5 w-3.5" /> 搜索
            </Button>
            {tab === 'sessions' && (
              <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeDeleted}
                  onChange={(e) => setIncludeDeleted(e.target.checked)}
                  className="rounded"
                />
                含已删除
              </label>
            )}
          </div>
        )}

        {tab === 'overview' && debug && <OverviewPanel debug={debug} />}
        {tab === 'sessions' && (
          <SessionsPanel
            sessions={sessions}
            total={sessionsTotal}
            page={sessionsPage}
            onPageChange={setSessionsPage}
            loading={loading}
            expandedSession={expandedSession}
            sessionLogs={sessionLogs}
            onExpand={loadSessionLogs}
            onDelete={handleDeleteSession}
            onRestore={handleRestoreSession}
            onPurgeDeleted={handlePurgeDeleted}
          />
        )}
        {tab === 'logs' && (
          <LogsPanel
            logs={logs}
            total={logsTotal}
            page={logsPage}
            onPageChange={setLogsPage}
            loading={loading}
            onDelete={handleDeleteLog}
          />
        )}
        {tab === 'phones' && (
          <PhonesPanel phones={phones} onToggle={handleTogglePhone} />
        )}
      </div>
    </div>
  )
}

function OverviewPanel({ debug }: { debug: DebugInfo }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {Object.entries(debug.tables)
          .filter(([k]) => !k.includes('error'))
          .map(([key, val]) => (
            <div key={key} className="rounded-xl border border-border bg-card p-4">
              <p className="text-xs text-muted-foreground">{key.replace(/_/g, ' ')}</p>
              <p className="text-2xl font-bold mt-1">{val}</p>
            </div>
          ))}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-border bg-card p-5">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" /> 服务状态
          </h3>
          <div className="space-y-2 text-sm">
            {Object.entries(debug.health).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-muted-foreground">{k}</span>
                <span className="font-mono">
                  {typeof v === 'boolean' ? (
                    <Badge variant={v ? 'default' : 'destructive'} className="text-xs">
                      {v ? '是' : '否'}
                    </Badge>
                  ) : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Server className="h-4 w-4 text-primary" /> 运行配置
          </h3>
          <div className="space-y-2 text-sm">
            {Object.entries(debug.config).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4">
                <span className="text-muted-foreground shrink-0">{k}</span>
                <span className="font-mono text-xs truncate text-right" title={v}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function SessionsPanel({
  sessions, total, page, onPageChange, loading, expandedSession, sessionLogs,
  onExpand, onDelete, onRestore, onPurgeDeleted,
}: {
  sessions: AdminSession[]
  total: number
  page: number
  onPageChange: (p: number) => void
  loading: boolean
  expandedSession: string | null
  sessionLogs: AdminLog[]
  onExpand: (sid: string) => void
  onDelete: (sid: string) => void
  onRestore: (sid: string) => void
  onPurgeDeleted: () => void
}) {
  const totalPages = Math.ceil(total / 30)

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-muted-foreground">共 {total} 条会话</p>
        <Button
          variant="outline"
          size="sm"
          onClick={onPurgeDeleted}
          className="gap-1.5 text-destructive hover:text-destructive"
        >
          <AlertTriangle className="h-3.5 w-3.5" />
          清除所有软删除
        </Button>
      </div>
      <div className="rounded-xl border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              {['', 'ID', '时间', '会话ID', '模式', '帧数', '识别结果', '置信度', '耗时(ms)', '状态', '操作'].map((h) => (
                <th key={h} className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={11} className="px-3 py-8 text-center text-muted-foreground">加载中...</td></tr>
            )}
            {!loading && sessions.length === 0 && (
              <tr><td colSpan={11} className="px-3 py-8 text-center text-muted-foreground">暂无数据</td></tr>
            )}
            {sessions.map((s, i) => (
              <>
                <tr
                  key={s.session_id}
                  className={`cursor-pointer hover:bg-muted/40 ${i % 2 === 0 ? 'bg-background' : 'bg-muted/20'} ${s.is_deleted ? 'opacity-50' : ''}`}
                  onClick={() => onExpand(s.session_id)}
                >
                  <td className="px-3 py-2 w-8">
                    {expandedSession === s.session_id
                      ? <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-muted-foreground">{s.id}</td>
                  <td className="px-3 py-2 text-xs whitespace-nowrap">
                    {s.created_at ? new Date(s.created_at).toLocaleString('zh-CN') : '—'}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">{s.session_id}</td>
                  <td className="px-3 py-2">
                    <Badge variant="outline" className="text-xs">{s.mode}</Badge>
                  </td>
                  <td className="px-3 py-2 text-center">{s.total_frames}</td>
                  <td className="px-3 py-2 text-xs font-medium">{formatModelName(s.final_model_name)}</td>
                  <td className="px-3 py-2 tabular-nums">
                    {s.final_confidence != null ? `${(s.final_confidence * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="px-3 py-2 tabular-nums">{s.total_processing_ms.toFixed(0)}</td>
                  <td className="px-3 py-2">
                    {s.is_deleted
                      ? <Badge variant="destructive" className="text-xs">已删除</Badge>
                      : <Badge variant="default" className="text-xs bg-green-600">正常</Badge>}
                  </td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center gap-1.5">
                      {s.is_deleted && (
                        <button
                          onClick={() => onRestore(s.session_id)}
                          className="text-xs text-primary hover:underline flex items-center gap-1"
                          title="恢复软删除"
                        >
                          <RotateCcw className="h-3.5 w-3.5" /> 恢复
                        </button>
                      )}
                      <button
                        onClick={() => onDelete(s.session_id)}
                        className="text-xs text-destructive hover:underline flex items-center gap-1"
                        title="物理删除（不可恢复）"
                      >
                        <Trash2 className="h-3.5 w-3.5" /> 删除
                      </button>
                    </div>
                  </td>
                </tr>
                {expandedSession === s.session_id && (
                  <tr key={`${s.session_id}-detail`}>
                    <td colSpan={11} className="bg-muted/30 px-6 py-3">
                      <p className="text-xs text-muted-foreground mb-2 font-medium">
                        该会话下共 {sessionLogs.length} 条帧记录：
                      </p>
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-muted-foreground">
                            {['帧号', '文件名', '检测数', '检测置信度', '分类结果', '分类置信度', '质量分', '最优帧', '耗时(ms)'].map((h) => (
                              <th key={h} className="px-2 py-1 text-left font-medium">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {sessionLogs.map((l) => (
                            <tr key={l.id} className={l.is_best_frame ? 'bg-primary/10 font-medium' : ''}>
                              <td className="px-2 py-1">{l.frame_index}</td>
                              <td className="px-2 py-1 max-w-[160px] truncate" title={l.filename}>{l.filename}</td>
                              <td className="px-2 py-1">{l.detection_count}</td>
                              <td className="px-2 py-1 tabular-nums">
                                {l.detection_confidence != null ? l.detection_confidence.toFixed(3) : '—'}
                              </td>
                              <td className="px-2 py-1">{formatModelName(l.classification_model_name)}</td>
                              <td className="px-2 py-1 tabular-nums">
                                {l.classification_confidence != null ? `${(l.classification_confidence * 100).toFixed(1)}%` : '—'}
                              </td>
                              <td className="px-2 py-1 tabular-nums">
                                {l.quality_score != null ? l.quality_score.toFixed(3) : '—'}
                              </td>
                              <td className="px-2 py-1">
                                {l.is_best_frame ? <Badge className="text-xs bg-amber-500">★ 最优</Badge> : '—'}
                              </td>
                              <td className="px-2 py-1 tabular-nums">{l.processing_time_ms.toFixed(0)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4">
          <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
            上一页
          </Button>
          <span className="text-sm text-muted-foreground">{page} / {totalPages}</span>
          <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
            下一页
          </Button>
        </div>
      )}
    </div>
  )
}

function LogsPanel({
  logs, total, page, onPageChange, loading, onDelete,
}: {
  logs: AdminLog[]
  total: number
  page: number
  onPageChange: (p: number) => void
  loading: boolean
  onDelete: (logId: number) => void
}) {
  const totalPages = Math.ceil(total / 50)

  return (
    <div>
      <p className="text-sm text-muted-foreground mb-3">共 {total} 条日志</p>
      <div className="rounded-xl border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              {['ID', '会话ID', '帧号', '文件名', '检测数', '分类结果', '分类置信度', '质量分', '最优帧', '耗时(ms)', '时间', '操作'].map((h) => (
                <th key={h} className="px-2 py-2 text-left text-xs font-medium text-muted-foreground">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={12} className="px-3 py-8 text-center text-muted-foreground">加载中...</td></tr>
            )}
            {!loading && logs.length === 0 && (
              <tr><td colSpan={12} className="px-3 py-8 text-center text-muted-foreground">暂无数据</td></tr>
            )}
            {logs.map((l, i) => (
              <tr key={l.id} className={i % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
                <td className="px-2 py-1.5 tabular-nums text-muted-foreground">{l.id}</td>
                <td className="px-2 py-1.5 font-mono text-xs">{l.session_id.slice(0, 10)}…</td>
                <td className="px-2 py-1.5 text-center">{l.frame_index}</td>
                <td className="px-2 py-1.5 text-xs max-w-[140px] truncate" title={l.filename}>{l.filename}</td>
                <td className="px-2 py-1.5 text-center">{l.detection_count}</td>
                <td className="px-2 py-1.5 text-xs">{formatModelName(l.classification_model_name)}</td>
                <td className="px-2 py-1.5 tabular-nums">
                  {l.classification_confidence != null ? `${(l.classification_confidence * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="px-2 py-1.5 tabular-nums">{l.quality_score != null ? l.quality_score.toFixed(3) : '—'}</td>
                <td className="px-2 py-1.5">
                  {l.is_best_frame ? <Badge className="text-xs bg-amber-500">★</Badge> : '—'}
                </td>
                <td className="px-2 py-1.5 tabular-nums">{l.processing_time_ms.toFixed(0)}</td>
                <td className="px-2 py-1.5 text-xs whitespace-nowrap">
                  {l.created_at ? new Date(l.created_at).toLocaleString('zh-CN') : '—'}
                </td>
                <td className="px-2 py-1.5">
                  <button
                    onClick={() => onDelete(l.id)}
                    className="text-muted-foreground hover:text-destructive"
                    title="物理删除"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4">
          <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
            上一页
          </Button>
          <span className="text-sm text-muted-foreground">{page} / {totalPages}</span>
          <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
            下一页
          </Button>
        </div>
      )}
    </div>
  )
}

function PhonesPanel({
  phones,
  onToggle,
}: {
  phones: PhoneModelItem[]
  onToggle: (id: number) => void
}) {
  return (
    <div>
      <p className="text-sm text-muted-foreground mb-3">共 {phones.length} 个型号</p>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {phones.map((p) => (
          <div key={p.id} className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-medium">{p.model_name || p.model_key}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {p.manufacturer} · {p.brand}
                </p>
              </div>
              <Badge variant={p.is_active ? 'default' : 'secondary'} className="text-xs">
                {p.is_active ? '启用' : '停用'}
              </Badge>
            </div>
            <div className="flex items-center justify-between mt-3">
              <p className="font-mono text-xs text-muted-foreground">
                key: {p.model_key}
              </p>
              <button
                onClick={() => onToggle(p.id)}
                className="flex items-center gap-1 text-xs text-primary hover:underline"
                title={p.is_active ? '停用该型号' : '启用该型号'}
              >
                <Power className="h-3.5 w-3.5" />
                {p.is_active ? '停用' : '启用'}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
