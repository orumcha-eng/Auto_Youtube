
import requests
import time

def test_pollinations_video():
    prompt = "A cute white bean character dancing, 3d render, high quality"
    
    # Attempt 1: GET request with model param (Common for pollinations)
    # They often redirect to the result
    url = f"https://image.pollinations.ai/prompt/{prompt}?model=seedance&width=512&height=512&nologo=true"
    
    print(f"Testing URL: {url}")
    
    try:
        resp = requests.get(url, stream=True)
        print(f"Status Code: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('Content-Type')}")
        
        if resp.status_code == 200:
            if "video" in resp.headers.get('Content-Type', ''):
                print("Success! It returned a video.")
                with open("test_pollinations.mp4", "wb") as f:
                     for chunk in resp.iter_content(chunk_size=1024):
                         f.write(chunk)
            elif "image" in resp.headers.get('Content-Type', ''):
                 print("Returned an IMAGE, not video. Maybe seedance is image-only via this endpoint?")
                 # Check if it's a GIF?
            else:
                print("Unknown content type.")
        else:
            print("Failed.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pollinations_video()
