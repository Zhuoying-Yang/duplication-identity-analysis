import pandas as pd
import torch

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = "/shared/ssd_30T/zhuoyingyang/models/Qwen3-VL-8B-Instruct"

PROMPT = """
Carefully inspect this generated robot manipulation video.

First identify the target object that the robot is trying to manipulate.

Then track that SAME physical target object through time.

I specifically want you to check for OBJECT IDENTITY FAILURES:

1. DUPLICATION:
   Does one target object spontaneously become two simultaneously visible
   physical objects at any point?

   The new object may be almost identical to the original, or may be a
   similar but slightly changed object.

2. DISAPPEARANCE:
   Does the target object spontaneously disappear without being physically
   occluded, placed inside something, or leaving the camera view?

Please describe what happens temporally.

In particular report:
- what the target object is
- how many target/target-like objects exist near the beginning
- whether a new target-like object appears later
- whether the original remains visible
- whether the robot later manipulates the original or the newly appearing object
- whether any unexplained disappearance happens

Do NOT force a binary answer.
Describe exactly what you observe.
"""


print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()


df = pd.read_csv("manifest.csv")

for _, r in df.iterrows():

    case = r["case"]
    path = r["video_path"]

    print("\n" + "=" * 100)
    print(case)
    print(path)
    print("=" * 100)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": path,
                    "fps": 8.0
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    # Qwen3-VL returns (video_tensor, video_metadata) pairs.
    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        videos = None
        video_metadata = None

    # Some qwen_vl_utils / transformers version combinations
    # return fps as a one-element list, while the processor
    # expects a scalar for a single video.
    if isinstance(video_kwargs.get("fps"), list):
        if len(video_kwargs["fps"]) == 1:
            video_kwargs["fps"] = video_kwargs["fps"][0]

    print("video_kwargs:", video_kwargs)
    print("video metadata:", video_metadata)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadata,
        padding=True,
        return_tensors="pt",
        **video_kwargs
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("\nQWEN ANSWER:")
    print(answer)
