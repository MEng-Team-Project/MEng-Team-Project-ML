import traffic_ml.bin.live_metadata as live_metadata

m3u8_path = "C:\\Users\\win8t\\OneDrive\\Desktop\\projects\\traffic-web\\server\\livestream\\"

playlist = live_metadata.load(f'{m3u8_path}output.m3u8')  # this could also be an absolute filename
print(playlist.segments)
print(playlist.target_duration)