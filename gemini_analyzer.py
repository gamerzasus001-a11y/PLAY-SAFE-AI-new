import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("ERROR: API Key not found! Check your .env file.")

client = genai.Client(api_key=API_KEY)

def get_gemini_coach_insight(video_path: str):
    """Uploads a video to Gemini for expert-level biomechanical analysis."""
    if not os.path.exists(video_path):
        return "Gemini Analysis: Video file not found for cloud processing."

    print(f"\n>> [Gemini AI] Uploading '{video_path}' for advanced reasoning review...")
    try:
        video_file = client.files.upload(file=video_path)
    except Exception as e:
        return f">> [Gemini AI] Upload error: {e}"
    
    print(">> [Gemini AI] Processing video on cloud servers...")
    try:
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
            
        if video_file.state.name == "FAILED":
            return ">> [Gemini AI] Processing failed on Google servers."
    except Exception as e:
        return f">> [Gemini AI] File state error: {e}"
        
    prompt = """
    You are an elite sports physiotherapist and lead biomechanics expert. Provide a comprehensive, professional analysis of this workout video:
    1. Posture & Spinal Alignment: Detail the spine position, head posture, and core engagement.
    2. Kinematic & Joint Deviations: Pinpoint any structural flaws, stress points, or improper joint angles.
    3. Elite Action Plan: Deliver a rigorous, multi-paragraph coaching summary with precise structural adjustments and safety recommendations.
    """
    
    try:
        # Using gemini-3.5-flash (the correct, highly supported current generation Flash model)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[video_file, prompt]
        )
        return response.text
    # Rock-solid exception block so API bugs never crash your main app
    except BaseException as e:
        return f">> [Gemini AI] Generation error (Server/Quota Issue): {e}"
    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass