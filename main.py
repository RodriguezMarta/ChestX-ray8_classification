from src.training.train import train_model

if __name__ == "__main__":
    csv_file = "data/metadata/Data_Entry_2017_v2020.csv"
    img_dir = "data/images/"
    output_model_path = "models/chestxray_supervised.pth"

    # Entrenar el modelo supervisado
    train_model(csv_file, img_dir, output_model_path)
