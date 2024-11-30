# Comprobamos si tenemos las imágenes y si no se tienen, se procede a la descarga
import os
import tarfile
tar_files_dir = '../data/zip_images/'

output_dir = '../data/'  

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener todos los archivos .tar.gz de la carpeta
tar_files = [f for f in os.listdir(tar_files_dir) if f.endswith('.tar.gz')]

# Descomprimir cada archivo .tar.gz
for tar_file in tar_files:
    tar_path = os.path.join(tar_files_dir, tar_file)
    print(f'Descomprimiendo {tar_path}...')
    
    # Abrir el archivo tar.gz y extraer su contenido
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(output_dir)  # Extraer todo en la carpeta de salida
    
    print(f'{tar_file} descomprimido en {output_dir}')

print("Todos los archivos .tar.gz han sido descomprimidos.")