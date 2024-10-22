import tkinter as tk

# Función que se ejecutará al hacer clic en el botón
def on_button_click():
    label.config(text="¡Hola, mundo!")

# Crear la ventana principal
root = tk.Tk()
root.title("Canvas-Camera")  # Título de la ventana
root.state('zoomed')  # Maximizar la ventana (Windows)

# Cambiar el icono de la ventana
root.iconbitmap("PáginaCanva\images\icono.ico")  # Cambia esto por la ruta de tu icono

# Cambiar el color de fondo a negro
root.configure(bg='#212121')  # Cambia '#000000' por el color que prefieras

# Crear una etiqueta
label = tk.Label(root, text="Haz clic en el botón", font=("Arial", 14), bg='#212121', fg='white')  # Cambiar el color de fondo de la etiqueta
label.pack(pady=20)  # Añadir margen vertical

# Crear un botón
button = tk.Button(root, text="Haz clic aquí", command=on_button_click)
button.pack(pady=10)  # Añadir margen vertical

# Iniciar el bucle principal
root.mainloop()
