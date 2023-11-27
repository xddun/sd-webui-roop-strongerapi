# encoding=utf-8
'''
Author: SpenserCai
Date: 2023-08-20 17:28:26
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-21 17:05:30
Description: file content
'''
import base64
import json

from fastapi import FastAPI, Body

from modules.api.models import *
from modules import scripts, shared
from modules.api import api
from modules import paths_internal
import gradio as gr
from PIL import Image
from scripts.faceswap import get_models
from scripts.swapper import UpscaleOptions, swap_face, ImageResult, upscale_image_and_face_restorer

from fastapi import FastAPI, HTTPException
import requests

from scripts.roop_logging import logger

question_prefix_main = """

Create a English NNN-word description on the topic: "TEXTGENDER".
Examples: 
"cyborg woman| with a visible detailed brain| muscles cable wires| detailed cyberpunk background with neon lights| biopunk| cybernetic| unreal engine| CGI | ultra detailed| 4k".
"video games icons, 2d icons, rpg skills icons, world of warcraft items icons, league of legends items icons, ability icon, fantasy, potions, spells, objects, flowers, gems, swords, axe, hammer, fire, ice, arcane, shiny object, graphic design, high contrast, artstation - -uplight --v 4",
"2 warrior princesses fight, Dynamic pose; Artgerm, Wlop, Greg Rutkowski; the perfect mix of Emily Ratajkowski, Ana de Armas, Kate Beckinsale, Kelly Brook and Adriana Lima as warrior princess; high detailed tanned skin; beautiful long hair, intricately detailed eyes; druidic leather vest; wielding an Axe; Attractive; Flames in background; Lumen Global Illumination, Lord of the Rings, Game of Thrones, Hyper-Realistic, Hyper-Detailed, 8k, --no watermarks --no cape --testp --ar 9:16 --v 4 --upbeta",
"highly detailed matte painting stylized three quarters portrait of an anthropomorphic rugged happy fox with sunglasses! head animal person, background blur bokeh! ! ; by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski --v 4",
"a photo of 8k ultra realistic archangel with 6 wings, full body, intricate purple and blue neon armor, ornate, cinematic lighting, trending on artstation, 4k, hyperrealistic, focused, high details, unreal engine 5, cinematic --ar 9:16 --s 1250 --q 2",
"helldog, cerberus, red dog, big dog, 3 heads, rotten skin, background gate of hell, background hell, character design, side view, full body view, full body, aggressive stil, mythic still, unreal engine, hyper-detailed",
"faeries in a bubble gum world , cinematic view, cinematic lighting, HD, 8k, unreal engine 5, full realistic, landscape, octane, high details, unreal engine 5, octane render, cryengine, volumetric lighting, cinematic, mood, dust, particles, particle effect, atmosphere, ray tracing, uhd, lighting rendered in Unreal Engine 5 and Cinema4k, Diffraction Grating, Ray Tracing Ambient Occlusion, Antialiasing, GLSL-Shaders, Post Process, Post Production, Cell Shading, Tone Mapping, photorealistic --v 4",
"bronze metal raven, magical automaton, intricate details --v 4",
"person = barbara palvin as undead necromancer lich. looks =dark straight hair, feminine figure, gorgeous, pretty face,beautiful body, wings, beautiful symmetrical glowing red eyes, two beautiful smooth silky legs:: pose = arms around body, Dragon with Massive Wings spread completely behind back:: shape = dance pose, curvy body, anatomically correct and fully showing symmetrical female chest, anatomically correct female torso and belly button, complete smooth pretty human arms and fingers, symmetrical face. clothing = revealing outfit, A pair of wearing a ivory breastplate and full-body-jewelry harness made from ice jewels :: environment = cemetary with frost aura ::details = highly detailed clothing,clear face that is fully shown, hyper quality style, top shelf jewelry with jewel arrays = dark fantasy, realistic, full female-body shot, both lower and, upper lips are completely down in clear detail, 3d printed:: artist = cgsociety, artgerm, trending on artstation, by victor titov. --video --s 3250 --seed 909101793 --ar 10:18 --q 1.5 --no amputees --no blur DOF --no fractals --v 3",
"two lovers as neural networks embracing, beautiful, intricate details, cinematic lighting, beautiful concept art, surreal, art station",
"realistic photograph of police nousr robot in modern city, cyberpunk, character design, detailed face, highly detailed, intricate details, symmetrical, smooth, digital 3d, hard surface, mechanical design, real-time, vfx, ultra hd, hdr"



After you create a description (not included in the word limit):
If in the description that you created a landscape described, then add "landscape, many details, a lot of detail" at the end.
If the description you created describes a portrait then add "focused, blured background, body, portret" to the end.
If the description you created describes a full-length person, then add "full-body, legs, arms" to the end.
If you are describing things that can have reflections, then add "reflection".
If you are describing a delicious food, please state as many ingredients and appearance as possible.
If you are describing an item or a particular object or thing, please state as many ingredients and appearance as possible.
If the item you are describing is unique to China, you must describe the material and appearance of the item.


Always! Add at the end one of the types of lighting that suits better to the end  in format: "Lighting: ". Here is the list: Spot Lighting, Rear light, Dark Light, Blinding Light, Candle Light, Concentrated Lighting, Twilight, Sunshine, Sunset, Lamp, Electric Arc, Moonlight, Neon Light, Night Light, Nuclear Light, Cinematic Light or similar.
Always! Add one of the styles that fits better to the end in format: "Style: ". Here is the list: Fantasy, Dark Fantasy, Abstraction, Ethereal, Weirdcore, Dreampunk, Daydreampunk, Science, Surrealism, Unrealistic, Surreal, Realistic, Photorealism, Classic, Retro, Retrowave, Vintage, Cyberpunk, Punk, Modern, Futuristic, Sci-fi, Alchemy, Alien, Aurora, Magic, Mystic, Marvel Comics, Anime, Cartoon, Manga, Kawaii, Pastel, Neon, Aesthetic, Miniature.
Very desirable. If some lighting suits the im.
After the description, a more suitable style and type of lighting should always be written.
Always! Add 5 additional parameters in the format: "Details: ". Here is a list of possible:: mood, dust, particles, particle effect, atmosphere, ray tracing, uhd, lighting, Diffraction Grating, Ray Tracing Ambient Occlusion, Antialiasing, GLSL-Shaders, Post Process, Post Production, Cell Shading, Tone Mapping and similar and any similar ones that will add the atmosphere to the description.
Always! You must output an English description, must not contain a Chinese character.

At the end it should be like this: "A Victorian-style chair with chrome and ornate decorations reflects a distorted image in the water on the ground. The intricate patterns on the chair add an air of sophistication and elegance to any room. Lighting: Candle Light. Style: Victorian. Details: Cell Shading, atmosphere, ray tracing."
"""

host_port = "http://localhost:7860"

prefabricate_prompt = "(realistic, masterpiece, best quality),<lora:add_detail:0.5>,"

prefabricate_negative_prompt = """paintings,sketches, (worst quality, low quality, normal quality:1.7), lowres, blurry, text, logo, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, strabismus, wrong finger, lowres, bad anatomy, bad hands, text,error,missing fingers,cropped, jpeg artifacts,signature,watermark, username, blurry, bad feet, (dark skin:1.1), fused girls, fushion, bad-hands-5, lowres, bad anatomy, bad hands, text, error, missing fingers,  cropped,  signature, watermark, username, blurry, (bad feet:1.1),, monochrome, jpeg artifacts, ugly, pregnant, vore, duplicate, morbid, mutilated, tran nsexual, hermaphrodite, long neck, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad proportions, malformed limbs, extra limbs, cloned face, disfigured, gross proportions, (missing arms:1.331), (missing legs:1.331), (extra arms:1.331), (extra legs:1.331), plump, bad legs, lowres, bad anatomy, bad hands, error, missing fingers, extra digit, fewer digits, (open month:2),(tooth:2),(teeth:2),(nsfw:2),"""

prefabricate_prompt_gender_gril = "(perfect middle breast:0.5),perfect figure,"

prefabricate_prompt_gender_boy = "muscle,Strength,Manhood,Virility,"


def get_chat_completion_content(question="领克有哪些型号", timeout=50):
    url_baichuan = "http://flyme-aigc.flyme.com/v1/chat/completions/Baichuan2-13B-Chat"
    url = "http://flyme-aigc.flyme.com/v1/chat/completions/Qwen-14B-Chat"
    headers = {
        'Content-Type': 'application/json'
    }

    data_baichuan = {
        "stream": 0,
        # "model_name": "Baichuan2-13B-Chat",
        "model_name": "--",
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.90,
        "top_k": 2,
        "peresence_penalty": 1,
        "repetition_penalty": 1.05,
        "messages": [
            {
                "content": question,
                "role": "user"
            }
        ]
    }
    data = {
        "stream": 0,
        "model_name": "--",
        "temperature": 0.3,
        "max_tokens": 2047,
        "top_p": 0.90,
        "top_k": 2,
        "peresence_penalty": 1,
        "repetition_penalty": 1.05,
        "stop": [],
        "messages": [
            {
                "content": question,
                "role": "user"
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'], response.status_code
        else:
            return None, 508
    except:
        try:
            response = requests.post(url_baichuan, headers=headers, json=data_baichuan, timeout=timeout)
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'], response.status_code
            else:
                return None, 508
        except:
            logger.info(f"get_chat_completion_content error, LLM error")
            return None, 508


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


def save_encoded_image(b64_image, output_path):
    with open(output_path, 'wb') as image_file:
        image_file.write(base64.b64decode(b64_image))


def roop_api(_: gr.Blocks, app: FastAPI):
    @app.get("/roop/models")
    async def roop_models():
        models = []
        for model in get_models():
            models.append(model.split("/")[-1])
        return {"models": models}

    @app.post("/roop/image")
    def roop_image(
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
    def image_upscale(
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

    @app.post("/base/generate_image")
    def generate_image(user_prompt: str = Body("None", title="user prompt"),
                       timeout: int = Body(20, title="call llm' timeout time(s)"),
                       enable_hr: bool = Body(True, title="enable hr"),
                       seed: int = Body(-1, title="seed"),
                       gender: str = Body("", title="gender"),
                       enable_LLM: bool = Body(True, title="enable LLM"),
                       batch_size: int = Body(1, title="batch_size,generate batch_size images "),
                       width: int = Body(384, title="images width "),
                       height: int = Body(512, title="images height "),
                       ):
        if timeout > 50:
            raise HTTPException(status_code=500,
                                detail=f'Error: timeout must <=50, llm model not is VIP ')
        if batch_size <= 0:
            raise HTTPException(status_code=500,
                                detail=f'Error: batch_size must >=1')
        if batch_size > 4:
            raise HTTPException(status_code=500,
                                detail=f'Error: batch_size must <=4')
        if width < 64 or width > 512:
            raise HTTPException(status_code=500,
                                detail=f'Error: width must 64<=width<=512')
        if height < 64 or height > 512:
            raise HTTPException(status_code=500,
                                detail=f'Error: height must 64<=height<=512')
        if len(user_prompt) > 500:
            raise HTTPException(status_code=507,
                                detail=f'Error: len(user_prompt)>500, user_prompt must be <=500')
        logger.info(f"batch_size {batch_size}")
        if gender != "" and enable_LLM is False:
            raise HTTPException(status_code=500,
                                detail=f'Error: Choosing to strengthen gender must relies on a large language model')
        if gender != "":
            if gender != "boy" and gender != "girl":
                raise HTTPException(status_code=500, detail=f'gender value:"boy" or "girl"')
        if user_prompt == "None":
            raise HTTPException(status_code=500, detail=f'Error: user_prompt is empty')
        txt2img_url = f"{host_port}/sdapi/v1/txt2img"
        logger.info(f"generate_image: user_prompt={user_prompt}, timeout={timeout}, seed={seed}, gender={gender}")
        if enable_LLM:
            question = question_prefix_main.replace("TEXT", user_prompt.replace("\n", "")).replace("NNN", "50")
            if gender == "":
                question = question.replace("GENDER", "")
            elif gender == "boy":
                question = question.replace("GENDER", ",男性写真照描述")
            elif gender == "girl":
                question = question.replace("GENDER", ",女性写真照描述")
            llm_source_prompt, status_code = get_chat_completion_content(question, timeout)
            logger.info(f"generate_image: llm_source_prompt={llm_source_prompt}, status_code={status_code}")
            if llm_source_prompt is None:
                raise HTTPException(status_code=501, detail=f'Error: {status_code} Error requesting llm model')
            else:
                llm_source_prompt = ''.join(char for char in llm_source_prompt if ord(char) < 128)
                if len(llm_source_prompt) < 1:
                    raise HTTPException(status_code=504, detail=f'Error: prompt is empty')
                if gender == "boy":
                    llm_source_prompt = "1boy," + llm_source_prompt + ",(muscle:0.5),(solo portrait:0.99),wearing clothes,"
                elif gender == "girl":
                    llm_source_prompt = "1girl," + llm_source_prompt + ",(perfect middle breast:0.5),(solo portrait:0.99),wearing clothes,"
        else:
            llm_source_prompt = user_prompt

        # 前面加上一些固定的文字
        prompt = llm_source_prompt + prefabricate_prompt
        prompt = prompt.replace("\n", "")

        negative_prompt = prefabricate_negative_prompt

        # 去除非ascii字符
        ascii_negative_prompt = ''.join(char for char in negative_prompt if ord(char) < 128)
        negative_prompt = ascii_negative_prompt

        logger.info(f"prompt:  {prompt}")
        logger.info(f"negative_prompt:  {negative_prompt}")

        # 生成图片
        payload = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'sampler_index': 'DPM++ 2M Karras',
            'seed': seed,
            'steps': 25,
            'width': width,
            'height': height,
            'cfg_scale': 7.5,
            "batch_size": batch_size,
            "denoising_strength": 0.7,
            "enable_hr": enable_hr,
            "hr_scale": 2.0,
            "hr_upscaler": "Latent",
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "save_images": False,
            "alwayson_scripts": {
                "ADetailer": {
                    "args": [
                        {
                            "ad_model": "face_yolov8n.pt"
                        },
                        {
                            "ad_model": "hand_yolov8n.pt"
                        }
                    ]
                },
            },
            "override_settings": {
                "CLIP_stop_at_last_layers": 2,
            }
        }

        try:
            response = requests.post(txt2img_url, data=json.dumps(payload))
        except:
            raise HTTPException(status_code=505, detail=f'Error: call sdapi error')

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=505, detail=f'Error: {response.status_code} call sdapi error')

    @app.post("/base/generate_image_and_swapface")
    def generate_image_and_swapface(user_prompt: str = Body("None", title="user prompt"),
                                    timeout: int = Body(20, title="call llm' timeout time(s)"),
                                    enable_hr: bool = Body(True, title="enable hr"),
                                    face_image_b64: str = Body("None", title="face image"),
                                    enable_hr_face: bool = Body(True, title="enable hr face"),
                                    seed: int = Body(-1, title="seed"),
                                    gender: str = Body("", title="gender"),
                                    enable_LLM: bool = Body(True, title="enable LLM"),
                                    batch_size: int = Body(1, title="batch_size,generate batch_size images "),
                                    width: int = Body(384, title="images width "),
                                    height: int = Body(512, title="images height "),
                                    ):
        if timeout > 50:
            raise HTTPException(status_code=500,
                                detail=f'Error: timeout must <=50, llm model not is VIP ')
        if batch_size <= 0:
            raise HTTPException(status_code=500,
                                detail=f'Error: batch_size must >=1')
        if batch_size > 4:
            raise HTTPException(status_code=500,
                                detail=f'Error: batch_size must <=4')
        if width < 64 or width > 512:
            raise HTTPException(status_code=500,
                                detail=f'Error: width must 64<=width<=512')
        if height < 64 or height > 512:
            raise HTTPException(status_code=500,
                                detail=f'Error: height must 64<=height<=512')
        if len(user_prompt) > 500:
            raise HTTPException(status_code=507,
                                detail=f'Error: len(user_prompt)>500, user_prompt must be <=500')
        logger.info(f"batch_size {batch_size}")
        if face_image_b64 == "None":
            raise HTTPException(status_code=500, detail=f'Error: face_image_b64 is empty')

        #############################下面内容直接拉上面那个函数
        if gender != "" and enable_LLM is False:
            raise HTTPException(status_code=500,
                                detail=f'Error: Choosing to strengthen gender must relies on a large language model')
        if gender != "":
            if gender != "boy" and gender != "girl":
                raise HTTPException(status_code=500, detail=f'gender value:"boy" or "girl"')
        if user_prompt == "None":
            raise HTTPException(status_code=500, detail=f'Error: user_prompt is empty')
        txt2img_url = f"{host_port}/sdapi/v1/txt2img"
        logger.info(f"generate_image: user_prompt={user_prompt}, timeout={timeout}, seed={seed}, gender={gender}")
        if enable_LLM:
            question = question_prefix_main.replace("TEXT", user_prompt.replace("\n", "")).replace("NNN", "50")
            if gender == "":
                question = question.replace("GENDER", "")
            elif gender == "boy":
                question = question.replace("GENDER", ",男性写真照描述")
            elif gender == "girl":
                question = question.replace("GENDER", ",女性写真照描述")
            llm_source_prompt, status_code = get_chat_completion_content(question, timeout)
            logger.info(f"generate_image: llm_source_prompt={llm_source_prompt}, status_code={status_code}")
            if llm_source_prompt is None:
                raise HTTPException(status_code=501, detail=f'Error: {status_code} Error requesting llm model')
            else:
                llm_source_prompt = ''.join(char for char in llm_source_prompt if ord(char) < 128)
                if len(llm_source_prompt) < 1:
                    raise HTTPException(status_code=504, detail=f'Error: prompt is empty')
                if gender == "boy":
                    llm_source_prompt = "1boy," + llm_source_prompt + ",(muscle:0.5),(solo portrait:0.99),wearing clothes,"
                elif gender == "girl":
                    llm_source_prompt = "1girl," + llm_source_prompt + ",(perfect middle breast:0.5),(solo portrait:0.99),wearing clothes,"
        else:
            llm_source_prompt = user_prompt

        # 前面加上一些固定的文字
        prompt = llm_source_prompt + prefabricate_prompt
        prompt = prompt.replace("\n", "")

        negative_prompt = prefabricate_negative_prompt

        # 去除非ascii字符
        ascii_negative_prompt = ''.join(char for char in negative_prompt if ord(char) < 128)
        negative_prompt = ascii_negative_prompt

        logger.info(f"prompt:  {prompt}")
        logger.info(f"negative_prompt:  {negative_prompt}")

        #############################上面内容直接拉上面那个函数

        logger.info(f"prompt:  {prompt}")
        logger.info(f"negative_prompt:  {negative_prompt}")
        if enable_hr_face:
            upscaler_name = "R-ESRGAN 2x+"
        else:
            upscaler_name = "None"
        # 生成图片
        payload = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'sampler_index': 'DPM++ 2M Karras',
            'seed': seed,
            'steps': 25,
            'width': width,
            'height': height,
            'cfg_scale': 7.5,
            "batch_size": batch_size,
            "denoising_strength": 0.7,
            "enable_hr": enable_hr,
            "hr_scale": 2.0,
            "hr_upscaler": "Latent",
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "save_images": False,
            "alwayson_scripts": {
                "ADetailer": {
                    "args": [
                        {
                            "ad_model": "face_yolov8n.pt"
                        },
                        {
                            "ad_model": "hand_yolov8n.pt"
                        }
                    ]
                },
                "roop": {
                    "args": [face_image_b64,
                             True,
                             "0",
                             "/home/xiedong/stable-diffusion-webui/models/roop/inswapper_128.onnx",
                             "CodeFormer", 1,
                             upscaler_name, 2, 1,
                             False,
                             True
                             ]
                }
            },
            "override_settings": {
                "CLIP_stop_at_last_layers": 2,
            }
        }
        try:
            response = requests.post(txt2img_url, data=json.dumps(payload))
        except:
            raise HTTPException(status_code=505, detail=f'Error: call sdapi error')

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=505, detail=f'Error: {response.status_code} call sdapi error')


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(roop_api)
except:
    pass

