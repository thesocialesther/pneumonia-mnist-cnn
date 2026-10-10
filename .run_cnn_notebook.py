import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent
os.chdir(project)
path = project / 'pnuemo.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
class Tee(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)
    def flush(self):
        sys.__stdout__.flush()
count = 0
for index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    count += 1
    print(f'\nExecuting notebook cell {index}...', flush=True)
    captured = Tee()
    plots = []
    def show():
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
            plots.append({'output_type':'display_data', 'metadata':{},
                          'data':{'image/png':base64.b64encode(buffer.getvalue()).decode('ascii')}})
            plt.close(fig)
    plt.show = show
    cell['execution_count'] = count
    cell['outputs'] = []
    failure = None
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(''.join(cell['source']), f'notebook cell {index}', 'exec'), namespace)
    except Exception as error:
        failure = error
    text = captured.getvalue()
    if text:
        cell['outputs'].append({'output_type':'stream','name':'stdout','text':text.splitlines(keepends=True)})
    cell['outputs'].extend(plots)
    if failure:
        cell['outputs'].append({'output_type':'error','ename':type(failure).__name__,
                                'evalue':str(failure),'traceback':traceback.format_exception(failure)})
    path.write_text(json.dumps(nb, indent=1)+'\n', encoding='utf-8')
    if failure:
        raise failure
print('\nAll notebook cells executed successfully; outputs and figures saved.', flush=True)
