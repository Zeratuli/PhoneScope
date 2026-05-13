export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface DetectionItem {
  bbox: BoundingBox
  confidence: number
  label: string
  crop_base64: string | null
}

export interface TopKItem {
  name: string
  confidence: number
}

export interface PhoneSpec {
  manufacturer: string
  brand: string
  model: string
  released: string
  screen: string
  processor: string
  ram: string
  storage: string
  rear_camera: string
  front_camera: string
  battery: string
  os: string
  dimensions: string
  weight: string
  colors: string
}

export interface ClassificationItem {
  model_name: string
  brand_name: string | null
  series_name: string | null
  display_name: string | null
  confidence: number
  top_k: TopKItem[]
  phone_spec: PhoneSpec | null
}

export interface ImageResult {
  image_id: string
  filename: string
  width: number
  height: number
  detections: DetectionItem[]
  classifications: ClassificationItem[]
  annotated_image_base64: string
  processing_time_ms: number
}

export interface AnalysisResponse {
  success: boolean
  result: ImageResult
}

export interface BatchTaskResponse {
  success: boolean
  task_id: string
  total_files: number
  message: string
}

export interface TaskStatusResponse {
  task_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  current_file: string | null
  results: ImageResult[] | null
  created_at: string
  error: string | null
}

export interface HealthResponse {
  status: string
  models_loaded: boolean
  version: string
  timestamp: string
}

export interface FileWithPreview {
  id: string
  file: File
  preview: string
  type: 'image' | 'video'
}

export interface FrameEvidence {
  frame_index: number
  filename: string
  width: number
  height: number
  detections: DetectionItem[]
  best_detection_index: number | null
  quality_score: number
  is_valid: boolean
  is_best: boolean
  annotated_image_base64: string
}

export interface FusionResult {
  session_id: string
  mode: string
  total_frames: number
  valid_frames: number
  best_frame_index: number | null
  frames: FrameEvidence[]
  final_model_name: string | null
  final_brand_name?: string | null
  final_series_name?: string | null
  final_display_name?: string | null
  final_confidence: number | null
  final_top_k: TopKItem[] | null
  final_phone_spec: PhoneSpec | null
  best_crop_base64: string | null
  processing_time_ms: number
}

export interface FusionResponse {
  success: boolean
  result: FusionResult
}

export type ModelType = 'swin' | 'mobilenet'
export type ProcessingMode =
  | 'single'
  | 'batch'
  | 'fusion_images'
  | 'fusion_video'

export interface LogItem {
  id: number
  session_id: string
  frame_index: number
  filename: string
  image_width: number
  image_height: number
  detection_count: number
  detection_confidence: number | null
  classification_model_name: string | null
  classification_confidence: number | null
  quality_score: number | null
  is_best_frame: boolean
  processing_time_ms: number
  created_at: string | null
}

export interface LogsResponse {
  total: number
  page: number
  size: number
  items: LogItem[]
}

export interface SessionItem {
  id: number
  session_id: string
  mode: string
  total_frames: number
  final_model_name: string | null
  final_confidence: number | null
  best_frame_index: number | null
  total_processing_ms: number
  created_at: string | null
}

export interface SessionsResponse {
  total: number
  page: number
  size: number
  items: SessionItem[]
}

export interface AdminDebugInfo {
  health: HealthResponse
  db_stats: StatsResponse
  routes: string[]
  config: Record<string, string>
  tables: Record<string, number>
}

export interface StatsResponse {
  total_detections: number
  total_sessions: number
  avg_detection_count: number
  top_models: { name: string; count: number }[]
  mode_counts: Record<string, number>
}

export interface NumberedTask {
  taskNumber: number
  taskId: string
  mode: ProcessingMode
  modelType: ModelType
  status: 'pending' | 'processing' | 'completed' | 'failed'
  summary: string
  createdAt: string
}
