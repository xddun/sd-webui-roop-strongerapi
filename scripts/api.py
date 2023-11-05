'''
Author: SpenserCai
Date: 2023-08-20 17:28:26
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-21 17:05:30
Description: file content
'''
from fastapi import FastAPI, Body

from modules.api.models import *
from modules import scripts, shared
from modules.api import api
from modules import paths_internal
import gradio as gr
from PIL import Image
from scripts.faceswap import get_models
from scripts.swapper import UpscaleOptions, swap_face, ImageResult, upscale_image_and_face_restorer


def get_face_restorer(str):
    for restorer in shared.face_restorers:
        if restorer.name() == str:
            return restorer
    return None


def get_full_model(model_name):
    models = get_models()
    for model in models:
        if model.split("/")[-1] == model_name:
            return model
    return None


def roop_api(_: gr.Blocks, app: FastAPI):
    @app.get("/roop/models")
    async def roop_models():
        models = []
        for model in get_models():
            models.append(model.split("/")[-1])
        return {"models": models}

    @app.post("/roop/image")
    async def roop_image(
            source_image: str = Body("", title="source face image"),
            target_image: str = Body("", title="target image"),
            face_index: list[int] = Body([0], title="face index"),
            scale: int = Body(1, title="scale"),
            upscale_visibility: float = Body(1, title="upscale visibility"),
            face_restorer: str = Body("None", title="face restorer"),
            restorer_visibility: float = Body(1, title="face restorer"),
            model: str = Body("inswapper_128.onnx", title="model"),
            upscaler_name: str = Body("None", title="upscaler name"),
    ):
        s_image = api.decode_base64_to_image(source_image)
        t_image = api.decode_base64_to_image(target_image)
        f_index = set(face_index)
        upscalerx = None
        if upscaler_name != "None":
            for upscaler in shared.sd_upscalers:
                if upscaler.name == upscaler_name:
                    upscalerx = upscaler
        up_options = UpscaleOptions(scale=scale,
                                    upscaler=upscalerx,
                                    upscale_visibility=upscale_visibility,
                                    face_restorer=get_face_restorer(face_restorer),
                                    restorer_visibility=restorer_visibility)
        use_model = get_full_model(model)
        if use_model is None:
            Exception("Model not found")
        result = swap_face(s_image, t_image, use_model, f_index, up_options)
        return {"image": api.encode_pil_to_base64(result.image())}

    # 增加只进行图像增强+人脸恢复的接口，不进行换脸
    @app.post("/roop/image_upscale")
    async def image_upscale(
            source_image: str = Body("", title="source face image"),
            upscaler_name: str = Body("None", title="upscaler name"),  # 增强器
            scale: int = Body(1, title="scale"),
            upscale_visibility: float = Body(1, title="upscale visibility"),
            face_restorer: str = Body("None", title="face restorer"),
            restorer_visibility: float = Body(1, title="face restorer"),
    ):
        s_image = api.decode_base64_to_image(source_image)
        upscalerx = None
        if upscaler_name != "None":
            for upscaler in shared.sd_upscalers:
                if upscaler.name == upscaler_name:
                    upscalerx = upscaler
        up_options = UpscaleOptions(scale=scale,
                                    upscaler=upscalerx,
                                    upscale_visibility=upscale_visibility,
                                    face_restorer=get_face_restorer(face_restorer),
                                    restorer_visibility=restorer_visibility)
        result = upscale_image_and_face_restorer(s_image, up_options)
        return {"image": api.encode_pil_to_base64(result.image())}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(roop_api)
except:
    pass
