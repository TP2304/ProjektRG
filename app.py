import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageUploadApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Upload and Recognition")
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        # Upload Image Button
        self.upload_image_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack()

        self.image_path = None
        self.image = None

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((800, 600), Image.Resampling.LANCZOS)  # Resizes the image
            self.image = ImageTk.PhotoImage(image)

            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor='nw', image=self.image)
            self.canvas.image = self.image  # Store reference to image

            messagebox.showinfo("Image Upload", "Image uploaded successfully!")
        else:
            messagebox.showwarning("No Image", "Please select an image to upload.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploadApp(root)
    root.mainloop()
