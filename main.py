import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class ObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Skin Vision")

        self.model = YOLO("best (1).onnx", task='detect')

        # Set window size and position
        window_width = 400
        window_height = 300
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width / 2) - (window_width / 2)
        y = (screen_height / 2) - (window_height / 2)
        master.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')

        # Create a frame for the title
        self.title_frame = tk.Frame(master, bg="light blue", width=window_width, height=50)
        self.title_frame.pack_propagate(False)  # Prevent frame from shrinking to fit contents
        self.title_frame.pack(side=tk.TOP, fill=tk.X)

        # Create label for the title
        self.title_label = tk.Label(self.title_frame, text="Skin Vision", font=("Helvetica", 20), bg="light blue")
        self.title_label.pack(fill=tk.BOTH, expand=True)

        # Create a frame for the buttons
        self.button_frame = tk.Frame(master, width=window_width, height=window_height - 50, bg="light gray")
        self.button_frame.pack_propagate(False)  # Prevent frame from shrinking to fit contents
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH)

        # Create buttons with custom styles
        self.button_style = ttk.Style()
        self.button_style.configure("TButton", font=("Helvetica", 12), background="light blue", relief="raised", borderwidth=0)

        self.image_button = ttk.Button(self.button_frame, text="Select Image", command=self.load_image, style="Round.TButton")
        self.image_button.pack(pady=10, padx=20, ipadx=20, ipady=10)

        self.camera_button = ttk.Button(self.button_frame, text="Capture from Camera", command=self.capture_from_camera, style="Round.TButton")
        self.camera_button.pack(pady=10, padx=20, ipadx=20, ipady=10)

        # Create label to display the image
        self.image_label = tk.Label(master)
        self.image_label.pack()

        # Create a custom style for round buttons
        self.button_style.configure("Round.TButton", borderradius=50)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = cv2.imread(file_path)
            results = self.model(image)
            output_image = self.draw_boxes_on_image(image, results[0].boxes)

            image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)

            self.image_label.config(image=photo)
            self.image_label.image = photo

    def capture_from_camera(self):
        cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
        ret, frame = cap.read()
        cap.release()

        results = self.model(frame)
        output_image = self.draw_boxes_on_image(frame, results[0].boxes)

        image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def draw_boxes_on_image(self, image, boxes):
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image

def main():
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()