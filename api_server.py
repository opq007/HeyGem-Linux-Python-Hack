# -*- coding: utf-8 -*-
"""
HeyGem 数字人 FastAPI 服务
提供带 Token 鉴权的数字人视频生成接口
"""

import os
import sys
import uuid
import time
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, List
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, HttpUrl, Field
import httpx

# 检查 Python 版本
if sys.version_info.major != 3 or sys.version_info.minor != 8:
    print("请使用 Python 3.8 版本运行此脚本")
    sys.exit(1)

# 导入 torch（必须在导入项目模块之前，以便设置 CUDA 设备）
try:
    import torch
except ImportError:
    print("警告: 未安装 torch，CUDA 设备选择功能将不可用")
    torch = None

# =============================================================================
# CUDA 显卡选择
# =============================================================================

def setup_cuda_device():
    """
    设置 CUDA 设备
    从环境变量 CUDA_DEVICE_ID 读取要使用的显卡 ID
    """
    cuda_device_id = os.getenv("CUDA_DEVICE_ID", "0")
    
    if torch is None:
        print("警告: torch 未安装，无法设置 CUDA 设备")
        return
    
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        return
    
    # 获取可用 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个可用 GPU")
    
    # 显示所有 GPU 信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    
    # 设置要使用的 GPU
    try:
        device_id = int(cuda_device_id)
        if device_id < 0 or device_id >= gpu_count:
            print(f"错误: CUDA_DEVICE_ID {device_id} 超出范围，可用范围: 0-{gpu_count-1}")
            print(f"将使用默认 GPU 0")
            device_id = 0
        
        # 设置 CUDA_VISIBLE_DEVICES 环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(f"已设置使用 GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        
        # 验证设置
        torch.cuda.empty_cache()
        current_device = torch.cuda.current_device()
        print(f"当前 CUDA 设备: {current_device}")
        
    except ValueError:
        print(f"错误: CUDA_DEVICE_ID '{cuda_device_id}' 不是有效的数字")
        print("将使用默认 GPU 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 在导入项目模块之前设置 CUDA 设备
setup_cuda_device()

# 导入项目模块
import service.trans_dh_service
from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger


# =============================================================================
# 配置和初始化
# =============================================================================

# 从环境变量获取鉴权 Token
AUTH_TOKEN = os.getenv("DIGITAL_HUMAN_AUTH_TOKEN", "")
if not AUTH_TOKEN:
    logger.warning("环境变量 DIGITAL_HUMAN_AUTH_TOKEN 未设置，将允许所有请求！")
    logger.warning("请设置 DIGITAL_HUMAN_AUTH_TOKEN 环境变量以启用鉴权")

# =============================================================================
# 生命周期管理和服务初始化函数
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    使用新的 lifespan 事件处理器替代弃用的 on_event
    """
    import asyncio
    
    # 启动时执行
    print("=" * 60)
    print("开始启动 HeyGem 数字人服务...")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("开始启动 HeyGem 数字人服务...")
    logger.info("=" * 60)
    
    try:
        print("正在调用 init_service...")
        logger.info("正在调用 init_service...")
        
        # 使用线程池执行初始化，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_service)
        
        print("init_service 执行完成")
        logger.info("init_service 执行完成")
        
        # 验证服务是否成功初始化
        if service_initialized:
            print("=" * 60)
            print("✓ 服务初始化成功，可以开始处理请求")
            print("=" * 60)
            logger.info("=" * 60)
            logger.info("✓ 服务初始化成功，可以开始处理请求")
            logger.info("=" * 60)
        else:
            print("=" * 60)
            print("✗ 服务初始化失败：服务未标记为已初始化")
            print("=" * 60)
            logger.error("=" * 60)
            logger.error("✗ 服务初始化失败：服务未标记为已初始化")
            logger.error("=" * 60)
            # 不再抛出异常，让应用继续启动但标记为未初始化
            
    except Exception as e:
        print("=" * 60)
        print(f"✗ 服务启动失败: {str(e)}")
        print("=" * 60)
        logger.error("=" * 60)
        logger.error(f"✗ 服务启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        # 不再抛出异常，让应用继续启动但标记为未初始化
        # 这样健康检查会返回正确的状态
    
    # 生成
    print("lifespan yield, 应用已启动")
    logger.info("lifespan yield, 应用已启动")
    yield
    
    # 关闭时执行
    print("HeyGem 数字人服务正在关闭...")
    logger.info("HeyGem 数字人服务正在关闭...")

def init_service():
    """
    初始化数字人服务
    
    会同步等待服务完全初始化完成，确保在应用启动前所有资源都已就绪
    """
    global digital_human_service, service_initialized
    
    print(f"init_service 被调用, service_initialized={service_initialized}")
    logger.info(f"init_service 被调用, service_initialized={service_initialized}")
    
    if service_initialized:
        logger.info("服务已经初始化，跳过重复初始化")
        return
    
    try:
        print("正在加载模型和初始化数字人服务...")
        logger.info("正在加载模型和初始化数字人服务...")
        print("这个过程可能需要几分钟时间，请耐心等待...")
        logger.info("这个过程可能需要几分钟时间，请耐心等待...")
        
        # 创建服务实例（这会触发模型加载）
        print("正在创建 TransDhTask 实例...")
        logger.info("正在创建 TransDhTask 实例...")
        digital_human_service = service.trans_dh_service.TransDhTask()
        print("TransDhTask 实例创建完成")
        logger.info("TransDhTask 实例创建完成")
        
        logger.info("服务实例创建完成，正在验证服务可用性...")
        
        # 动态等待服务真正初始化完成（而不是固定等待 10 秒）
        max_wait_time = 300  # 最长等待 5 分钟
        check_interval = 2   # 每 2 秒检查一次
        waited_time = 0
        
        print(f"开始验证服务可用性，最长等待 {max_wait_time} 秒...")
        logger.info(f"开始验证服务可用性，最长等待 {max_wait_time} 秒...")
        
        while waited_time < max_wait_time:
            try:
                # 尝试访问服务内部状态来判断是否初始化完成
                # 如果服务有 task_dic 属性且可访问，说明初始化基本完成
                if hasattr(digital_human_service, 'task_dic'):
                    # 尝试访问 task_dic 来验证服务可用
                    _ = digital_human_service.task_dic
                    print(f"服务验证通过，初始化完成 (等待了 {waited_time} 秒)")
                    logger.info("服务验证通过，初始化完成")
                    service_initialized = True
                    break
                
            except Exception as check_error:
                print(f"服务验证中... ({waited_time}/{max_wait_time}秒): {str(check_error)}")
                logger.debug(f"服务验证中... ({waited_time}/{max_wait_time}秒): {str(check_error)}")
            
            time.sleep(check_interval)
            waited_time += check_interval
        
        # 检查是否超时
        if not service_initialized:
            error_msg = f"服务初始化超时（等待了 {max_wait_time} 秒），请检查系统资源和模型文件"
            print(error_msg)
            raise TimeoutError(error_msg)
        
        print("✓ 数字人服务初始化成功！")
        logger.info("✓ 数字人服务初始化成功！")
        
    except Exception as e:
        print(f"✗ 数字人服务初始化失败: {str(e)}")
        logger.error(f"✗ 数字人服务初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        service_initialized = False
        raise

# =============================================================================
# FastAPI 应用配置
# =============================================================================

# FastAPI 主应用（根路径）
# 注意：lifespan 必须挂到主应用上，因为 uvicorn 启动的是主应用
# 子应用的 lifespan 在 mount 时不会自动触发
app = FastAPI(
    title="HeyGem 数字人 API",
    description="提供数字人视频生成服务",
    version="1.0.0",
    lifespan=lifespan
)

# FastAPI 子应用（挂载到 /heygem）
heygem_app = FastAPI(
    title="HeyGem 数字人 API",
    description="提供数字人视频生成服务",
    version="1.0.0"
)

# CORS 中间件（应用到两个应用）
for fastapi_app in [app, heygem_app]:
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Token 鉴权
security = HTTPBearer(auto_error=False)

# =============================================================================
# 数据模型
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: TaskStatus
    message: str
    result_url: Optional[str] = None

class DigitalHumanRequest(BaseModel):
    """数字人生成请求模型（支持网络 URL 和本地文件路径）"""
    audio_url: str = Field(..., description="音频文件的 URL 地址或本地文件路径（支持 http/https 网络地址和本地绝对/相对路径）")
    video_url: str = Field(..., description="视频文件的 URL 地址或本地文件路径（支持 http/https 网络地址和本地绝对/相对路径）")
    watermark: bool = Field(False, description="是否添加水印")
    digital_auth: bool = Field(False, description="是否添加数字人标识")

class DigitalHumanResponse(BaseModel):
    """数字人生成响应模型"""
    task_id: str
    status: TaskStatus
    message: str
    result_video_url: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str  # 可以是 "ok", "ready", "initializing", "error" 等
    service: str
    version: str
    initialized: bool

# =============================================================================
# 全局变量和工具函数
# =============================================================================

# 任务存储（生产环境建议使用 Redis 或数据库）
tasks = {}
# 数字人服务实例
digital_human_service = None
service_initialized = False

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """
    验证 Token 鉴权
    
    Args:
        credentials: HTTP Bearer Token 凭证
        
    Returns:
        bool: 鉴权是否通过
        
    Raises:
        HTTPException: 鉴权失败
    """
    # 如果未设置 Token，允许所有请求（开发模式）
    if not AUTH_TOKEN:
        return True
    
    # 如果未提供 Token，拒绝访问
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="未提供鉴权 Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证 Token
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="无效的鉴权 Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True

def is_url(path: str) -> bool:
    """
    判断路径是否为网络 URL
    
    Args:
        path: 文件路径或 URL
        
    Returns:
        bool: 如果是 http/https URL 返回 True，否则返回 False
    """
    return path.lower().startswith(('http://', 'https://'))

async def resolve_file_path(file_path: str, save_dir: str, file_type: str) -> str:
    """
    解析文件路径，支持网络 URL 和本地文件路径
    
    Args:
        file_path: 文件路径（可以是 http/https URL 或本地路径）
        save_dir: 保存目录（用于下载网络文件）
        file_type: 文件类型（用于日志，如 "音频" 或 "视频"）
        
    Returns:
        str: 最终的本地文件路径
        
    Raises:
        HTTPException: 文件处理失败
    """
    try:
        # 如果是网络 URL，下载文件
        if is_url(file_path):
            logger.info(f"检测到网络 {file_type} URL: {file_path}")
            
            # 从 URL 中提取文件名
            filename = os.path.basename(file_path)
            if not filename or '.' not in filename:
                # 如果无法提取文件名，使用默认名称
                filename = f"{file_type.lower()}_downloaded.wav" if file_type == "音频" else f"{file_type.lower()}_downloaded.mp4"
            
            save_path = os.path.join(save_dir, filename)
            await download_file_from_url(file_path, save_path)
            logger.info(f"{file_type}文件下载成功: {save_path}")
            return save_path
        
        # 如果是本地路径
        else:
            logger.info(f"检测到本地 {file_type} 路径: {file_path}")
            
            # 检查路径是否存在
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"{file_type}文件不存在: {file_path}"
                )
            
            # 检查是否为文件
            if not os.path.isfile(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"{file_type}路径不是有效的文件: {file_path}"
                )
            
            # 返回绝对路径
            abs_path = os.path.abspath(file_path)
            logger.info(f"{file_type}文件路径解析成功: {abs_path}")
            return abs_path
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{file_type}文件路径解析失败: {file_path}, 错误: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"{file_type}文件路径解析失败: {str(e)}"
        )

async def download_file_from_url(url: str, save_path: str) -> str:
    """
    从 URL 下载文件
    
    Args:
        url: 文件 URL
        save_path: 保存路径
        
    Returns:
        str: 保存的文件路径
        
    Raises:
        HTTPException: 下载失败
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存文件
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"文件下载成功: {url} -> {save_path}")
            return save_path
            
    except Exception as e:
        logger.error(f"文件下载失败: {url}, 错误: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"文件下载失败: {str(e)}"
        )

def get_video_info(video_path: str) -> tuple:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        tuple: (width, height, fps)
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

# =============================================================================
# 自定义视频写入函数（用于异步处理）
# =============================================================================

def write_video_async(
    output_imgs_queue,
    temp_dir,
    result_dir,
    work_id,
    audio_path,
    result_queue,
    width,
    height,
    fps,
    watermark_switch=0,
    digital_auth=0,
):
    """
    异步视频写入函数
    """
    import cv2
    import subprocess
    
    output_mp4 = os.path.join(temp_dir, f"{work_id}-t.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_path = os.path.join(result_dir, f"{work_id}-r.mp4")
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    
    try:
        while True:
            state, reason, value_ = output_imgs_queue.get()
            
            if type(state) == bool and state == True:
                logger.info(f"视频帧队列处理已结束: {work_id}")
                video_write.release()
                break
            elif type(state) == bool and state == False:
                logger.error(f"任务视频帧队列异常: {work_id}, 原因: {reason}")
                result_queue.put([False, reason])
                return
            else:
                for result_img in value_:
                    video_write.write(result_img)
        
        if video_write is not None:
            video_write.release()
        
        # 使用 ffmpeg 合并音频和视频
        command = f"ffmpeg -loglevel warning -y -i {audio_path} -i {output_mp4} -c:a aac -c:v libx264 -crf 15 -strict -2 {result_path}"
        subprocess.call(command, shell=True)
        
        logger.info(f"视频生成完成: {result_path}")
        result_queue.put([True, result_path])
        
    except Exception as e:
        logger.error(f"视频写入异常: {work_id}, 错误: {str(e)}")
        result_queue.put([False, str(e)])

# 替换原有的 write_video 函数
service.trans_dh_service.write_video = write_video_async

# =============================================================================
# 启动时初始化
# =============================================================================

# =============================================================================
# API 接口
# =============================================================================

@heygem_app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    status = "ready" if service_initialized else "initializing"
    return HealthResponse(
        status=status,
        service="HeyGem Digital Human API",
        version="1.0.0",
        initialized=service_initialized
    )

@heygem_app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    status = "ready" if service_initialized else "initializing"
    return HealthResponse(
        status=status,
        service="HeyGem Digital Human API",
        version="1.0.0",
        initialized=service_initialized
    )

@heygem_app.get("/api/v1/gpu/info")
async def get_gpu_info(auth_verified: bool = Depends(verify_token)):
    """
    获取 GPU 信息
    
    返回:
    - cuda_available: CUDA 是否可用
    - gpu_count: 可用 GPU 数量
    - gpus: 所有 GPU 的详细信息列表
    - current_device: 当前使用的 GPU 设备 ID
    """
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "current_device": None
    }
    
    if torch is not None and torch.cuda.is_available():
        gpu_info["cuda_available"] = True
        gpu_info["gpu_count"] = torch.cuda.device_count()
        
        # 获取所有 GPU 信息
        for i in range(gpu_info["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory_total = gpu_props.total_memory / (1024**3)
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            gpu_memory_free = gpu_memory_total - gpu_memory_allocated
            
            gpu_info["gpus"].append({
                "id": i,
                "name": gpu_name,
                "memory_total_gb": round(gpu_memory_total, 2),
                "memory_allocated_gb": round(gpu_memory_allocated, 2),
                "memory_reserved_gb": round(gpu_memory_reserved, 2),
                "memory_free_gb": round(gpu_memory_free, 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count
            })
        
        # 获取当前设备
        gpu_info["current_device"] = torch.cuda.current_device()
        
        # 显示当前设置的 CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            gpu_info["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    
    return gpu_info

@heygem_app.post("/api/v1/digital-human/generate", response_model=DigitalHumanResponse)
async def generate_digital_human(
    request: DigitalHumanRequest,
    background_tasks: BackgroundTasks,
    auth_verified: bool = Depends(verify_token)
):
    """
    生成数字人视频（支持网络 URL 和本地文件路径）
    
    参数:
    - audio_url: 音频文件的 URL 地址或本地文件路径
      - 支持 http:// 和 https:// 网络地址（会自动下载）
      - 支持本地绝对路径或相对路径
    - video_url: 视频文件的 URL 地址或本地文件路径
      - 支持 http:// 和 https:// 网络地址（会自动下载）
      - 支持本地绝对路径或相对路径
    - watermark: 是否添加水印（默认 False）
    - digital_auth: 是否添加数字人标识（默认 False）
    
    返回:
    - task_id: 任务 ID
    - status: 任务状态
    - message: 消息
    - result_video_url: 结果视频 URL（任务完成后）
    - error: 错误信息（如果失败）
    
    示例:
    - 网络地址: {"audio_url": "https://example.com/audio.wav", "video_url": "https://example.com/video.mp4"}
    - 本地路径: {"audio_url": "/path/to/audio.wav", "video_url": "/path/to/video.mp4"}
    - 相对路径: {"audio_url": "example/audio.wav", "video_url": "example/video.mp4"}
    """
    # 检查服务是否初始化
    if not service_initialized:
        raise HTTPException(
            status_code=503,
            detail="服务尚未初始化完成，请稍后再试"
        )
    
    # 生成任务 ID
    task_id = str(uuid.uuid4())
    
    # 创建临时目录
    temp_dir = os.path.join(GlobalConfig.instance().temp_dir, task_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 解析音频和视频文件路径（自动识别网络 URL 或本地路径）
        audio_path = await resolve_file_path(request.audio_url, temp_dir, "音频")
        video_path = await resolve_file_path(request.video_url, temp_dir, "视频")
        
        # 初始化任务状态
        tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "message": "任务已提交，等待处理",
            "result_path": None,
            "error": None,
            "audio_path": audio_path,
            "video_path": video_path,
            "temp_dir": temp_dir,
            "watermark": request.watermark,
            "digital_auth": request.digital_auth
        }
        
        # 添加后台任务处理
        background_tasks.add_task(
            process_digital_human_task,
            task_id,
            audio_path,
            video_path,
            temp_dir,
            request.watermark,
            request.digital_auth
        )
        
        # 立即返回任务ID
        result_url = f"/heygem/api/v1/tasks/{task_id}"
        
        return DigitalHumanResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="任务已提交，正在后台处理中",
            result_video_url=result_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数字人生成任务提交失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 更新任务状态
        tasks[task_id] = {
            "status": TaskStatus.FAILED,
            "message": "任务提交失败",
            "result_path": None,
            "error": str(e)
        }
        
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return DigitalHumanResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            message="任务提交失败",
            error=str(e)
        )

async def process_digital_human_task(
    task_id: str,
    audio_path: str,
    video_path: str,
    temp_dir: str,
    watermark: bool,
    digital_auth: bool
):
    """
    后台处理数字人视频生成任务
    
    这个函数在后台线程中执行，不会阻塞 HTTP 响应
    """
    try:
        # 更新任务状态为处理中
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        tasks[task_id]["message"] = "正在生成数字人视频"
        logger.info(f"开始处理任务 {task_id}")
        
        # 获取视频信息
        width, height, fps = get_video_info(video_path)
        
        # 调用数字人服务
        digital_human_service.task_dic[task_id] = ""
        digital_human_service.work(
            audio_path,
            video_path,
            task_id,
            1 if watermark else 0,  # watermark_switch
            1 if digital_auth else 0,  # digital_auth
            0,  # 其他参数
            0
        )
        
        # 等待结果
        max_wait_time = 600  # 最长等待 10 分钟
        start_time = time.time()
        
        while True:
            if task_id in digital_human_service.task_dic:
                task_info = digital_human_service.task_dic[task_id]
                
                # 检查任务是否失败（task_info 为 False 或包含错误信息）
                if task_info is False:
                    logger.error(f"任务 {task_id} 处理失败（False 状态）")
                    tasks[task_id]["status"] = TaskStatus.FAILED
                    tasks[task_id]["message"] = "数字人视频生成失败"
                    tasks[task_id]["error"] = "后台处理异常"
                    return
                
                if task_info and len(task_info) >= 1:
                    # 检查是否有错误信息
                    if len(task_info) >= 2 and task_info[1] and isinstance(task_info[1], str) and task_info[1].startswith("[Error"):
                        error_msg = task_info[1]
                        logger.error(f"任务 {task_id} 处理失败: {error_msg}")
                        tasks[task_id]["status"] = TaskStatus.FAILED
                        tasks[task_id]["message"] = "数字人视频生成失败"
                        tasks[task_id]["error"] = error_msg
                        return
                
                # 检查是否有结果路径
                if task_info and len(task_info) >= 3:
                    result_path = task_info[2]
                    if result_path and os.path.exists(result_path):
                        # 移动结果文件到结果目录
                        final_result_dir = os.path.join("result", task_id)
                        os.makedirs(final_result_dir, exist_ok=True)
                        
                        final_result_path = os.path.join(
                            final_result_dir,
                            os.path.basename(result_path)
                        )
                        shutil.move(result_path, final_result_path)
                        
                        # 更新任务状态为完成
                        tasks[task_id]["status"] = TaskStatus.COMPLETED
                        tasks[task_id]["message"] = "数字人视频生成完成"
                        tasks[task_id]["result_path"] = final_result_path
                        tasks[task_id]["error"] = None
                        
                        # 清理临时文件
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        
                        logger.info(f"任务 {task_id} 处理成功: {final_result_path}")
                        return
            
            # 检查超时
            if time.time() - start_time > max_wait_time:
                logger.error(f"任务 {task_id} 处理超时")
                tasks[task_id]["status"] = TaskStatus.FAILED
                tasks[task_id]["message"] = "任务处理超时"
                tasks[task_id]["error"] = "处理时间超过 10 分钟"
                return
            
            # 短暂等待
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"任务 {task_id} 处理异常: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 更新任务状态
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["message"] = "数字人视频生成失败"
        tasks[task_id]["error"] = str(e)
        
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

@heygem_app.post("/api/v1/digital-human/upload")
async def upload_files(
    files: List[UploadFile] = File(..., description="上传的文件列表（音频和/或视频）"),
    auth_verified: bool = Depends(verify_token)
):
    """
    上传文件（音频和/或视频）
    
    参数:
    - files: 文件列表，支持上传音频和视频文件
      - 支持的音频格式: .wav, .mp3, .m4a, .aac
      - 支持的视频格式: .mp4, .avi, .mov, .mkv
    
    返回:
    - 上传成功的文件信息列表，包含:
      - filename: 文件名
      - file_type: 文件类型（audio/video）
      - size: 文件大小（字节）
      - path: 保存路径
      - url: 访问 URL
    """
    # 定义支持的文件类型
    audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    
    # 确保 output 目录存在
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    uploaded_files = []
    
    try:
        for file in files:
            # 获取文件扩展名
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            # 判断文件类型
            if file_ext in audio_extensions:
                file_type = "audio"
            elif file_ext in video_extensions:
                file_type = "video"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {file.filename}。支持的格式: 音频 {audio_extensions}, 视频 {video_extensions}"
                )
            
            # 生成唯一文件名（避免覆盖）
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{file.filename}"
            save_path = os.path.join(output_dir, unique_filename)
            
            # 保存文件
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # 获取文件大小
            file_size = os.path.getsize(save_path)
            
            # 生成访问 URL
            file_url = f"/api/v1/files/{unique_filename}"
            
            uploaded_files.append({
                "filename": file.filename,
                "saved_as": unique_filename,
                "file_type": file_type,
                "size": file_size,
                "path": save_path,
                "url": file_url
            })
            
            logger.info(f"文件上传成功: {file.filename} -> {save_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "message": f"成功上传 {len(uploaded_files)} 个文件",
            "files": uploaded_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败: {str(e)}"
        )

@heygem_app.get("/api/v1/files/{filename}")
async def get_file(
    filename: str,
    auth_verified: bool = Depends(verify_token)
):
    """
    获取上传的文件
    
    参数:
    - filename: 文件名（包含时间戳前缀）
    
    返回:
    - 文件内容
    """
    # 安全检查：防止路径遍历攻击
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=400,
            detail="无效的文件名"
        )
    
    file_path = os.path.join(os.getcwd(), "output", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="文件不存在"
        )
    
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=400,
            detail="不是有效的文件"
        )
    
    # 根据文件扩展名确定媒体类型
    file_ext = os.path.splitext(filename)[1].lower()
    media_type = "application/octet-stream"
    
    if file_ext in {'.wav'}:
        media_type = "audio/wav"
    elif file_ext in {'.mp3'}:
        media_type = "audio/mpeg"
    elif file_ext in {'.m4a'}:
        media_type = "audio/mp4"
    elif file_ext in {'.aac'}:
        media_type = "audio/aac"
    elif file_ext in {'.mp4'}:
        media_type = "video/mp4"
    elif file_ext in {'.avi'}:
        media_type = "video/x-msvideo"
    elif file_ext in {'.mov'}:
        media_type = "video/quicktime"
    elif file_ext in {'.mkv'}:
        media_type = "video/x-matroska"
    
    # 提取原始文件名（去掉时间戳前缀）
    original_filename = filename.split('_', 1)[1] if '_' in filename else filename
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=original_filename
    )

@heygem_app.get("/api/v1/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
    auth_verified: bool = Depends(verify_token)
):
    """
    获取任务结果视频
    
    参数:
    - task_id: 任务 ID
    
    返回:
    - 视频文件
    """
    # 检查任务是否存在
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail="任务不存在"
        )
    
    task_info = tasks[task_id]
    
    # 检查任务状态
    if task_info["status"] != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"任务未完成，当前状态: {task_info['status']}"
        )
    
    result_path = task_info["result_path"]
    
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(
            status_code=404,
            detail="结果文件不存在"
        )
    
    # 返回视频文件
    return FileResponse(
        path=result_path,
        media_type="video/mp4",
        filename=f"digital_human_{task_id}.mp4"
    )

@heygem_app.get("/api/v1/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    auth_verified: bool = Depends(verify_token)
):
    """
    获取任务状态
    
    参数:
    - task_id: 任务 ID
    
    返回:
    - 任务状态信息
    """
    # 检查任务是否存在
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail="任务不存在"
        )
    
    task_info = tasks[task_id]
    
    result_url = None
    if task_info["status"] == TaskStatus.COMPLETED and task_info["result_path"]:
        result_url = f"/api/v1/tasks/{task_id}/result"
    
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "message": task_info["message"],
        "result_video_url": result_url,
        "error": task_info["error"]
    }

# =============================================================================
# 挂载子应用到 /heygem 路径
# =============================================================================

app.mount("/heygem", heygem_app)

@app.get("/", response_model=HealthResponse)
async def app_root():
    """主应用根路径 - 重定向到健康检查"""
    return HealthResponse(
        status="ok",
        service="HeyGem Digital Human API",
        version="1.0.0",
        initialized=service_initialized
    )

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    cuda_device_id = os.getenv("CUDA_DEVICE_ID", "0")
    
    logger.info("=" * 60)
    logger.info("HeyGem 数字人 FastAPI 服务")
    logger.info("=" * 60)
    logger.info(f"服务地址: http://{host}:{port}")
    logger.info(f"鉴权 Token: {'已启用' if AUTH_TOKEN else '未启用（允许所有请求）'}")
    logger.info(f"CUDA 设备 ID: {cuda_device_id}")
    
    if torch is not None and torch.cuda.is_available():
        logger.info(f"当前使用 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 总内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        logger.warning("CUDA 不可用，将使用 CPU")
    
    logger.info("=" * 60)
    logger.info("API 接口:")
    logger.info("  GET  /                               - 主应用健康检查")
    logger.info("  GET  /heygem/                        - 子应用健康检查")
    logger.info("  GET  /heygem/health                  - 健康检查")
    logger.info("  GET  /heygem/docs                    - API 文档 (Swagger)")
    logger.info("  GET  /heygem/redoc                   - API 文档 (ReDoc)")
    logger.info("  GET  /heygem/openapi.json            - OpenAPI 规范")
    logger.info("  GET  /heygem/api/v1/gpu/info         - 查看 GPU 信息")
    logger.info("  POST /heygem/api/v1/digital-human/generate - 生成数字人视频 (URL 方式)")
    logger.info("  POST /heygem/api/v1/digital-human/upload   - 上传文件（音频/视频）")
    logger.info("  GET  /heygem/api/v1/files/{filename}       - 下载上传的文件")
    logger.info("  GET  /heygem/api/v1/tasks/{task_id}        - 查询任务状态")
    logger.info("  GET  /heygem/api/v1/tasks/{task_id}/result - 下载结果视频")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )