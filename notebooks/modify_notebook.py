import json

notebook_path = (
    'C:\\Users\\Preet\\ML-Final-Project\\ML_Final_Project (1).ipynb'
)

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = json.load(f)

    new_markdown_cell = {
        'cell_type': 'markdown',
        'id': 'gemini-generated-note-data-placeholder',
        'metadata': {},
        'source': [
            '**Note on Data:** The original data files for this project '
            '(`ML case Study.csv`, `Colleges.csv`, `cities.csv`) were not '
            'available. This notebook has been set up to run with '
            '**automatically generated placeholder data** that matches the '
            'expected schema. While the code demonstrates the full machine '
            'learning workflow, the analytical insights and model performance '
            'metrics are based on this synthetic data and should not be '
            'interpreted as findings from real-world data.'
        ]
    }

    if not notebook_content['cells'][0].get('id') == \
       'gemini-generated-note-data-placeholder':
        notebook_content['cells'].insert(0, new_markdown_cell)

    modified_cell_count = 0
    for cell in notebook_content['cells']:
        is_target_cell = (
            cell.get('id') == 'f4ec1f61-3006-4fa5-a58d-9452bb32f2b3' or
            (cell.get('cell_type') == 'code' and any(
                'pd.read_csv("C:\\\\Users\\\\preet\\\\Desktop' in s
                for s in cell.get('source', [])))
        )
        if is_target_cell:
            cell['source'] = [
                '# Read CSV files into Dataframes using relative paths\n',
                '\n',
                'df = pd.read_csv("data/ML case Study.csv")\n',
                'college = pd.read_csv("data/Colleges.csv")\n',
                'cities = pd.read_csv("data/cities.csv")'
            ]
            modified_cell_count += 1
            break

    if modified_cell_count == 0:
        print(
            "Warning: Data loading cell with hardcoded Desktop paths "
            "not found or modified."
        )

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1)

    print("Notebook modified successfully.")

except Exception as e:
    print(f"Error modifying notebook: {e}")
