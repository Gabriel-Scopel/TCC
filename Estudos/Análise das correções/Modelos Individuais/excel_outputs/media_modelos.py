import os
import pandas as pd

def calcular_medias_excel(diretorio_excel="excel_outputs"):
    """
    Lê todos os arquivos .xlsx do diretório especificado e calcula
    a média das notas dadas pelo professor e pelo modelo.
    """
    if not os.path.exists(diretorio_excel):
        print(f"O diretório '{diretorio_excel}' não existe.")
        return

    arquivos_excel = [f for f in os.listdir(diretorio_excel) if f.endswith('.xlsx')]

    if not arquivos_excel:
        print(f"Nenhum arquivo .xlsx encontrado em '{diretorio_excel}'.")
        return

    print("Médias encontradas por arquivo:\n")
    
    for arquivo in arquivos_excel:
        caminho_arquivo = os.path.join(diretorio_excel, arquivo)
        try:
            df = pd.read_excel(caminho_arquivo)

            media_professor = df["nota dada pelo professor"].mean()
            media_modelo = df["nota dada pelo modelo"].mean()

            print(f"Arquivo: {arquivo}")
            print(f"  Média do professor: {media_professor:.2f}")
            print(f"  Média do modelo:    {media_modelo:.2f}\n")
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

# Executa a função
calcular_medias_excel()
