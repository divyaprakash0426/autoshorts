
import sys
import os
import subprocess

def run_zenity():
    try:
        # Check if zenity exists
        subprocess.check_call(["which", "zenity"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Run zenity file selection
        result = subprocess.check_output(
            ["zenity", "--file-selection", "--title=Select Video File", "--file-filter=*.mp4 *.mkv *.mov *.avi *.webm"],
            stderr=subprocess.DEVNULL
        )
        print(result.decode("utf-8").strip())
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def run_tkinter():
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            print(file_path)
        root.destroy()
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    # Try zenity first on Linux
    if sys.platform.startswith("linux"):
        if run_zenity():
            sys.exit(0)
            
    # Fallback to tkinter
    if not run_tkinter():
        sys.exit(1)
