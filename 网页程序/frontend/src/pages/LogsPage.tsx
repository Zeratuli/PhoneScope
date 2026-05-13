import { useEffect, useState } from 'react'
import { getSessions, getLogStats, deleteSession } from '@/services/api'
import type { SessionItem, StatsResponse } from '@/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Trash2, RefreshCw } from 'lucide-react'
import { toast } from 'sonner'

const MODE_LABEL: Record<string, string> = {
  all: '全部模式',
  single: '单图',
  batch: '批量',
  fusion: '融合',
}

const MODE_COLOR: Record<string, string> = {
  single: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  fusion: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  batch: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
}

const FILTER_MODES = ['all', 'single', 'batch', 'fusion'] as const

function formatModelName(name: string | null) {
  if (!name) return '—'
  const labels: Record<string, string> = {
    HUAWEI_NOVA_10: 'HUAWEI nova 10',
    REDMI_K80_Pro: 'Redmi K80 Pro',
    iPhone_13: 'iPhone 13',
  }
  return labels[name] || name.replace(/_/g, ' ')
}

export default function LogsPage() {
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [dbError, setDbError] = useState(false)
  const [filterMode, setFilterMode] = useState('')
  const [filterModel, setFilterModel] = useState('')
  const PAGE_SIZE = 20

  const load = async () => {
    setLoading(true)
    setDbError(false)
    try {
      const [sessionsRes, statsRes] = await Promise.all([
        getSessions({
          page,
          size: PAGE_SIZE,
          mode: filterMode || undefined,
          model_name: filterModel || undefined,
        }),
        getLogStats(),
      ])
      setSessions(sessionsRes.items)
      setTotal(sessionsRes.total)
      setStats(statsRes)
    } catch {
      setDbError(true)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [page, filterMode, filterModel])

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSession(sessionId)
      toast.success('已删除该检测记录')
      load()
    } catch {
      toast.error('删除失败')
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="mx-auto max-w-7xl px-4 pt-24 pb-16 sm:px-6 lg:px-8">
      {dbError && (
        <div className="mb-4 rounded-lg border border-yellow-500/40 bg-yellow-500/10 p-4 text-sm text-yellow-600 dark:text-yellow-400">
          数据库未连接（MySQL 未启动）。请运行 <code className="font-mono bg-black/10 px-1 rounded">docker compose -p phonescope up -d</code> 启动 MySQL，然后重启后端。
        </div>
      )}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">数据管理</h1>
          <p className="text-muted-foreground mt-1">查看、筛选和管理所有检测记录（每次检测为一条）</p>
        </div>
        <Button variant="outline" size="sm" onClick={load} className="gap-2">
          <RefreshCw className="h-4 w-4" /> 刷新
        </Button>
      </div>

      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            { label: '检测次数', value: stats.total_sessions },
            { label: '总帧数', value: stats.total_detections },
            { label: '平均检测数', value: stats.avg_detection_count },
            { label: '最多识别型号', value: formatModelName(stats.top_models[0]?.name || null) },
          ].map((s) => (
            <div key={s.label} className="rounded-xl border border-border bg-card p-4">
              <p className="text-xs text-muted-foreground">{s.label}</p>
              <p className="text-2xl font-bold mt-1 truncate">{s.value}</p>
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-wrap gap-3 mb-4">
        <div className="flex flex-wrap gap-2 rounded-xl border border-border bg-card p-1">
          {FILTER_MODES.map((mode) => {
            const isActive = (mode === 'all' ? '' : mode) === filterMode
            return (
              <button
                key={mode}
                onClick={() => {
                  setFilterMode(mode === 'all' ? '' : mode)
                  setPage(1)
                }}
                className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-primary text-primary-foreground shadow-sm'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                }`}
              >
                {MODE_LABEL[mode]}
              </button>
            )
          })}
        </div>
        <input
          type="text"
          placeholder="按型号筛选..."
          value={filterModel}
          onChange={(e) => { setFilterModel(e.target.value); setPage(1) }}
          className="rounded-lg border border-border bg-card px-3 py-1.5 text-sm w-48"
        />
        <span className="text-sm text-muted-foreground self-center">共 {total} 条</span>
      </div>

      <div className="rounded-xl border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              {['#', '时间', '会话ID', '模式', '帧数', '识别结果', '置信度', '总耗时(ms)', '操作'].map((h) => (
                <th key={h} className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={9} className="px-3 py-8 text-center text-muted-foreground">
                  加载中...
                </td>
              </tr>
            )}
            {!loading && sessions.length === 0 && (
              <tr>
                <td colSpan={9} className="px-3 py-8 text-center text-muted-foreground">
                  暂无数据
                </td>
              </tr>
            )}
            {sessions.map((s, i) => (
              <tr
                key={s.id}
                className={i % 2 === 0 ? 'bg-background' : 'bg-muted/20'}
              >
                <td className="px-3 py-2 tabular-nums text-muted-foreground">{s.id}</td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">
                  {s.created_at ? new Date(s.created_at).toLocaleString('zh-CN') : '—'}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                  {s.session_id.slice(0, 10)}…
                </td>
                <td className="px-3 py-2">
                  <Badge variant="outline" className={`text-xs ${MODE_COLOR[s.mode] || ''}`}>
                    {MODE_LABEL[s.mode] || s.mode}
                  </Badge>
                </td>
                <td className="px-3 py-2 text-center">{s.total_frames}</td>
                <td className="px-3 py-2 text-xs font-medium">
                  {formatModelName(s.final_model_name)}
                </td>
                <td className="px-3 py-2 tabular-nums">
                  {s.final_confidence != null
                    ? `${(s.final_confidence * 100).toFixed(1)}%`
                    : '—'}
                </td>
                <td className="px-3 py-2 tabular-nums">
                  {s.total_processing_ms.toFixed(0)}
                </td>
                <td className="px-3 py-2">
                  <button
                    onClick={() => handleDelete(s.session_id)}
                    className="text-muted-foreground hover:text-destructive transition-colors"
                    title="删除该检测记录"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            上一页
          </Button>
          <span className="text-sm text-muted-foreground">
            {page} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            下一页
          </Button>
        </div>
      )}
    </div>
  )
}
