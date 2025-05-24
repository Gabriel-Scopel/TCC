import json
import pandas as pd
import os

def process_json_to_excel(json_file_path):
    """
    Reads a JSON file, extracts relevant grading information,
    and saves it to an Excel file.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo {json_file_path} não foi encontrado. Pulando.")
        return
    except json.JSONDecodeError:
        print(f"Erro: Não foi possível decodificar o JSON do arquivo {json_file_path}. Pulando.")
        return

    extracted_data = []
    
    # Determine the model grade key based on the filename
    model_grade_key = None
    if "correcao_use" in json_file_path.lower():
        model_grade_key = "use_grade"
    elif "deepseek" in json_file_path.lower():
        model_grade_key = "dpseek_grade"
    elif "correcao_bert" in json_file_path.lower():
        model_grade_key = "bert_grade"
    elif "correcao_elmo" in json_file_path.lower():
        model_grade_key = "elmo_grade"
    else:
        print(f"Aviso: Não foi possível determinar a chave da nota do modelo para {json_file_path}. Pulando este arquivo.")
        return

    # Define which files have the nested 'responses_students' structure
    # and use "grade" for the professor's grade.
    nested_structure_files = [
        "deepseeken.json", "deepseekes.json", "deepseekpt.json",
        "correcao_berten.json", "correcao_bertes.json",
        "correcao_bertpt.json" # <--- IMPORTANT FIX: correcao_bertPT.json also has nested structure
    ]
    
    # Normalize the filename for comparison (lowercase)
    current_file_normalized = os.path.basename(json_file_path).lower()
    
    requires_nested_iteration = current_file_normalized in nested_structure_files

    for entry in data:
        if requires_nested_iteration:
            # For files with nested 'responses_students' (deepseek, all bert files)
            if "responses_students" in entry:
                for response in entry["responses_students"]:
                    question_number = response.get("number_question")
                    # In nested structures, professor's grade is consistently "grade"
                    original_grade = response.get("grade") 
                    model_grade = response.get(model_grade_key)

                    # Convert grades to float, handle potential errors
                    try:
                        original_grade = float(original_grade) if original_grade is not None else None
                    except (ValueError, TypeError):
                        original_grade = None

                    try:
                        model_grade = float(model_grade) if model_grade is not None else None
                    except (ValueError, TypeError):
                        model_grade = None

                    extracted_data.append({
                        "numero da questão": question_number,
                        "nota dada pelo professor": original_grade,
                        "nota dada pelo modelo": model_grade
                    })
            else:
                print(f"Aviso: O arquivo {json_file_path} foi identificado como tendo estrutura aninhada, mas 'responses_students' não foi encontrado no topo do nível de uma entrada. Verifique a estrutura do JSON.")
        else:
            # For files with direct student response entries (use, elmo)
            question_number = entry.get("number_question")
            
            # In direct structures, professor's grade is "original_grade"
            original_grade = entry.get("original_grade") 
            model_grade = entry.get(model_grade_key)

            # Convert grades to float, handle potential errors
            try:
                original_grade = float(original_grade) if original_grade is not None else None
            except (ValueError, TypeError):
                original_grade = None

            try:
                model_grade = float(model_grade) if model_grade is not None else None
            except (ValueError, TypeError):
                model_grade = None

            extracted_data.append({
                "numero da questão": question_number,
                "nota dada pelo professor": original_grade,
                "nota dada pelo modelo": model_grade
            })

    if not extracted_data:
        print(f"Nenhum dado para extrair de {json_file_path}. O arquivo Excel não será gerado.")
        return

    df = pd.DataFrame(extracted_data)

    # Create output directory if it doesn't exist
    output_dir = "excel_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Generate Excel filename
    base_filename = os.path.basename(json_file_path)
    excel_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}.xlsx")

    try:
        df.to_excel(excel_filename, index=False)
        print(f"Dados exportados com sucesso para {excel_filename}")
    except Exception as e:
        print(f"Erro ao exportar para Excel para {json_file_path}: {e}")

# List of all your JSON files that are actually present
json_files = [
    "correcao_usePT.json",
    "correcao_use_EN.json", 
    "correcao_use_ES.json", 
    "deepseekEN.json",
    "deepseekES.json",
    "deepseekPT.json",
    "correcao_bertEN.json",
    "correcao_bertES.json",
    "correcao_bertPT.json", # Now correctly handled as nested
    "correcao_elmo_EN.json",
    "correcao_elmo_ES.json",
    "correcao_elmo_PT.json"
]

# Process each JSON file
for file in json_files:
    process_json_to_excel(file)