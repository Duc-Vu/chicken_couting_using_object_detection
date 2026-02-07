import yt_dlp

url = "https://www.youtube.com/shorts/nPooWEdTWPM"

ydl_opts = {
    "format": "bestvideo+bestaudio/best",
    "outtmpl": "%(title)s.%(ext)s"
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
 