## 🐍 Python

### Coding Style

<details>
<summary>📏 Python Sytle Guide</summary>

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/#basics-of-commenting-code)

<br>
</details>

<details>
<summary>📏 Documentation</summary>

```py
# Module

"""A one-line summary

Detailed descriptions for the module or program.
You may include 'how to run this' or 'usage of functions/classes'

"""

# Function
def function_name(args):
    """Function Description shortly

    More details for this class...
    More details for this class...

    Args:
        arg_name: description

    Returns:
        what to return
    
    Raises:
        error_type: why we get this error
    """

# Class
class ClassName:
    """A one-line summary

    More details for this class...
    More details for this class...

    Attributes:
        attrib_name: description
    """
```

<br>
</details>

<details>
<summary>📏 Indentations and Spaces</summary>

```py
# Examples: Parentheses --------------------------------
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

meal = (spam,
        beans)

foo = long_function_name(
    var_one, var_two, var_three,
    var_four)

foo = long_function_name(
    var_one, var_two, var_three,
    var_four
)

# String ------------------------------------------------
# One Tab (= 4 spaces)
long_string = """This is fine if your use case can accept
    extraneous leading spaces."""

# Use parentheses
long_string = ("And this is fine if you cannot accept\n" +
               "extraneous leading spaces.")
long_string = ("And this too is fine if you cannot accept\n"
               "extraneous leading spaces.")

# textwarp
import textwrap
long_string = textwrap.dedent("""\
    This is also fine, because textwrap.dedent()
    will collapse common leading spaces in each line.""")
```

<br>
</details>

<details>
<summary>📏 Function signature</summary>

```py
def my_method(
    self,
    first_var: int,
    second_var: Foo,
    third_var: Bar | None,
) -> int:

# spaces around `=` if the argument have type annotation & default value
def func(a: int = 0) -> int:
```

<br>
</details>

<details>
<summary>📏 Type Annotation</summary>

- `var: type = value` format
- `typing` module can be used

```py
# Variables
path: str = '/home/winterbloooom/foo.txt'
paths: list = [path1, path2, path3]

# Functions
def show_paths(paths: list, max_num: int = 3) -> str:
    return 'done'

# With `typing` module
from typing import List, Dict
food: List[str] = ['banana', 'apple']
students: Dict[str, int] = {'eungi': 100, 'winterbloooom': 99}
```

- References</summary>
  - [파이썬 타입 어노테이션/힌트 (Blog)](https://www.daleseo.com/python-type-annotations/)
  - [typing 모듈로 타입 표시하기 (Blog)](https://www.daleseo.com/python-typing/)

<br>
</details>

<details>
<summary>📏 Naming Convention</summary>

- Package / module - `package_name` , `module_name`
  - DO NOT use dashes(`-`)
- Function - `function_name`
- Variable
  - Global Constant - `GLOBAL_CONSTANT_NAME`
  - others - `var_name`
- Class - `ClassName`
- Exception - `ExceptionName`

Here's a guideline from [Gudio](https://en.wikipedia.org/wiki/Guido_van_Rossum)

|Type|	Public|	Internal|
|---|---|---|
|Packages               |`lower_with_under`   |                   |
|Modules                |`lower_with_under`     |`_lower_with_under`  |
|Classes                |`CapWords`           |`_CapWords`          |
|Exceptions             |`CapWords`           |               	|
|Functions              |`lower_with_under()` |`_lower_with_under()`|
|Global/Class Constants |`CAPS_WITH_UNDER`    |`_CAPS_WITH_UNDER`   |
|Global/Class Variables |`lower_with_under`   |`_lower_with_under`  |
|Instance Variables     |`lower_with_under`   |`_lower_with_under` (protected)|
|Method Names           |`lower_with_under()` |`_lower_with_under()` (protected)|
|Function/Method Parameters|`lower_with_under`|                   |
|Local Variables        |`lower_with_under`   |                   |

<br>
</details>

<details>
<summary>📏 Black & isort formatting</summary>

**[ Formatting with Black and isort ]**

- [Black](https://black.readthedocs.io/en/stable/index.html) for Python code formatting
- [isort](https://pycqa.github.io/isort/) for Python import sorting

**[ Method 1. VSCode extensions ]**

- [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) (by Microsoft)
- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) (by Microsoft)

 1. `command + shift + p`
 2. `Preferences: Open User Settings (JSON)`
 3. Insert code blow
   
```json
"[python]": {
    "diffEditor.ignoreTrimWhitespace": false,
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
},
"isort.args":["--profile", "black"],
```

**[ Method 2. Commandline ]**

- Installation**
  ```bash
  pip install black
  pip install isort
  ```
- Usage 1: command
  ```bash
  black <file or path>
  isort <file or path>
  ```
- Usage 2: with `pyproject.toml` configuration file
  - Make this file in the directory where the `.gitignore` exists.
    ```
    [tool.black]
    line-length = 100
    target-version = ['py39']
    exclude = '''
      \.git
      \DIR_OR_FILE_NAME
    '''
    
    [tool.isort]
    profile = "black"
    multi_line_output = 3
    use_parentheses = true
    line_length = 100
    skip = [".gitignore"]
    ```
  - then run commands below.
    ```bash
    black --config pyproject.toml <PATH>
    isort --settings-path pyproject.toml <PATH>
    ```

**[ Use with Pre-commit ]**
- Installation
  ```bash
  pip install pre-commit
  ```
- pre-commit configuration file
  - Make a file named `.pre-commit-config.yaml` in the directory where the `.gitignore` exist.
    ```yaml
    repos:
      - repo: https://github.com/PyCQA/isort
        rev: 5.10.1
        hooks:
          - id: isort
    
      - repo: https://github.com/ambv/black
        rev: 22.6.0
        hooks:
          - id: black
    ```
- Make pre-commit hook
  ```bash
  pre-commit install
  ```
- Commit
  ```bash
  git commit -am "pre-commit test"
  ```

<br>
</details>

<details>
<summary>📏 String Formatting</summary>

```py
# 천 단위 콤마 표시
print(f"{value:,}")
# 천 단위 콤마 표시 + 소숫점 (소숫점 앞 5자리, 뒤 2자리)
print(f"{value:5,.2f}")

# Scientific Notation (지수 표현)
print("{value:.2e}") # 1234567.89 -> 1.23e+06
print("{value:.2e}") # 0.0000001234 -> 1.23e-07
```

<br>
</details>

### For your smart codes

<details>
<summary>📌 Check the type or value of the function arguments</summary>

```py
if not isinstance(argument, (type1, type2, ...)):
    # Preprocess the argument

assert isinstance(argument, type1), f"Error message"
		# If fasle, Python occurs an AssertionError

# Example
def function_name(arg1, arg2):
    print(isinstance(arg1, str))
    assert isinstance(arg2, bool), f"""The type of 'arg2' is not matched. It should be {bool.__name__}, not {type(arg2).__name__}."""
```

<br>
</details>

<details>
<summary>📌 Configuration - OmegaConf</summary>

👉 **Basic Usage**
```py
from omegaconf import DictConfig

# yaml -> DictConfig
conf = OmegaConf.load('source/example.yaml')
# DictConfig -> yaml
print(OmegaConf.to_yaml(conf))

# Access
conf.dataset.name
conf['dataset']['name']

# Default Values
conf.get('missing_key', 'default_value')

# Merge configs
conf = OmegaConf.merge(base_cfg, model_cfg, optimizer_cfg, dataset_cfg) # each params are DictConfig types

# Convert to primitive container (dict)
primitive = OmegaConf.to_container(conf) # to_container(conf, resolve=True)
```

👉 **Resolvers**
- `oc.env`: environment variables
- `oc.create`: make new DictConfig
```yaml
user: ${oc.env:USER}
```

<br>
</details>

<details>
<summary>📌 Set the root directory</summary>

프로젝트 폴더 내에서 `from`, `import` 문을 사용해야 할 때 헷갈리는 경우가 있다. 
루트 디렉토리를 설정하면 `from 폴더1_이름.폴더2_이름 import 파일_이름` 식으로 사용이 쉽다.

- Choice 1: `pyrootutils.setup_root()`
    ```py
    # .git 이 있는 곳을 root로 지정
    import pyrootutils
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git"],
        pythonpath=True,
        dotenv=True,
    )
    ```
- Choice 2: `sys.path.insert()`
    ```py
    # os.path.dirname(__file__) : 현 파일이 있는 디렉토리 경로
    # sys.path.insert(0, [PATH]): [PATH]를 환경변수에 등록
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    ```

<br>
</details>

<details>
<summary>📌 Skip warning messages</summary>

```py
import warnings
warnings.simplefilter("ignore", UserWarning)
```

<br>
</details>

<details>
<summary>📌 Progress bar</summary>

```py
from tqdm import tqdm

for item in tqdm(my_list, desc="description")
for idx, item in enumerate(tqdm(my_list, desc='description'))
```

- References
  - [Reference1](https://skillmemory.tistory.com/entry/tqdm-%EC%82%AC%EC%9A%A9%EB%B2%95-python-%EC%A7%84%ED%96%89%EB%A5%A0-%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4%EB%B0%94)

<br>
</details>


<details>
<summary>📌 Parsing the arguments</summary>

```py
import argparse

parser = argparse.ArgumentParser(description="Description of this project")
parser.add_argument("--arg_name", type=int, default=None, help="description of this argument")
args = parser.parse_args()
```

- Description & default value of the argument
    ```py
    parser.add_argument("--arg_name", default=None, help="description of this argument")
    ```
- Define the names of the argument
    ```py
    parser.add_argument("--arg_name", "-n")
    ```
- Specify the type (e.g., string)
    ```py
    parser.add_argument("--arg_name", type=str)
    ```
- Specify the options
    ```py
    parser.add_argument("--arg_name", choices=[1, 2, 3])
    parser.add_argument("--arg_name", choices=range(0, 100))
    ```
- Actions
    - (1) store (default): store the value to the argument
    - (2) append: when you want to store multiple values as an list
        - e.g., `--arg_name 1 --arg_name "12", --arg_name False` -> `[1, "12", False]`
    - (3) store_true: store true
    ```py
    parser.add_argument('--arg_name', action='store_true')
    # [Wrong] parser.add_argument('--test', type=bool) -> if `--test False`, it also save True!
    # just `python main.py --arg_name`. If this argument not mentioned, False is stored.
    ```
- Specify the number of values
    - `N`: read N values (e.g., `--arg_name "spring" "winter"`)
    - `*`: read multiple values (e.g., `--arg_name 1 2 3 4`)
    - `+`: read at least one value
    - etc...
    ```py
    parser.add_argument('--arg_name', nargs='2') 
    ```
- Change the variable name to store the value
    ```py
    parser.add_argument("--arg_name", dest="arg_new_name")
    ```
- Positional (you must pass the value)
    - There isn't `-` before the name of the argument
    - You can just pass the value without the name (e.g., `python example.py "happy"`), just keep the sequence of positional arguments
    - If you want to change optional to positional, `parser.add_argument("--arg_name", required=True)``
    ```py
    parser.add_argument("arg_name")
    ```
- the number of arguments: `len(sys.argv)`
- Print the help: `parser.print_help()`

<br>
</details>


### Useful modules & functions

🌱 **Running arguments**

<details>
<summary>Python running arguements</summary>

- `-m`: run python module directly
  ```
  project/
  │── mypackage/
  │   │── __init__.py
  │   │── myscript.py
  │── main.py
  ```
  - You can run the `myscript.py` with `python -m mypackage.myscript` rather than `python mypackage/myscript` (it may occur import error)
- `-u`

<br>
</details>

🌱 **Get information about current status**

<details>
<summary>Date & Time in text</summary>

```py
import datetime
datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
# e.g., 24_02_16-17_26_20
```
<br>
</details>

<details>
<summary>Current running file/directory</summary>

```python
import os
# file name
f_name = os.path.abspath(__file__) # absolute path
f_name = os.path.realpath(__file__) # relateive path
# directory name
os.path.dirname(f_name)
```
<br>
</details>

<details>
<summary>Name of function</summary>

```python
import sys
sys._getframe(1).f_code.co_name # 현재 함수
sys._getframe(2).f_code.co_name # 이를 호출한 함수
```
<br>
</details>

<details>
<summary>Print the file name, line number, and function name</summary>

```python
import inspect
cf = inspect.currentframe()
print(f'\nFile "{cf.f_code.co_filename}", line {cf.f_lineno}, in {cf.f_code.co_name}')
# e.g., File "/home/eungi/4D-Humans/hmr2/datasets/__init__.py", line 68, in __init__
```
<br>
</details>

<details>
<summary>Attributes of Object</summary>

```py
hasattr(obj, 'age') # obj라는 개체에 'age'라는 속성이 있으면 True
getattr(obj, 'age', 'No age attribute') # obj라는 개체에 'age'라는 속성의 값을 가져오고, 없으면 세 번째 텍스트 출력
setattr(obj, 'age', 25) # obj라는 개체에 'age'라는 속성을 25로 추가/변경
```
<br>
</details>

<br>

🌱 **File/Directory Paths**

<details>
<summary>Existance of File/Directory</summary>

```python
import os
os.path.exist(PATH)
```
<br>
</details>

<details>
<summary>Compose file paths</summary>

```python
import os
path = '/home/data/my_dataset'
file_name = 'image_list.txt'
os.path.join(path, file_name)
```
<br>
</details>

<details>
<summary>Files in the directory </summary>

```python
import os
# file names
list_of_files = os.listdir('PATH_OF_DIR') # list
# file paths
list_of_paths = [os.path.join('DIR_PATH', fname) for fname in list_of_files]
```
<br>
</details>

<details>
<summary>List of files with condition</summary>

```py
import glob
file_list = glob.glob("*.jpg")
```
<br>
</details>

<br>

🌱 **Control**

<details>
<summary>Turn off the program here</summary>

```python
import sys; sys.exit()
```
<br>
</details>

<details>
<summary>pdb debugger</summary>

- import: `import pdb`
- break point: `pdb.set_trace()`
  - `n` to execute next line
  - `c` to continue (next break point)
  - `q` to quite
  - `s` to step into
- References
  - [pytorch 디버깅 함수 (Blog)](https://powerofsummary.tistory.com/166)
<br>
</details>

<br>

🌱 **File Handling**

<details>
<summary>text 파일 읽/쓰기</summary>

```py
with open("foo.txt", "r") as f:
    lines = f.readlines()

with open("foo.txt", "w") as f:
    f.write("Life is too short, you need python")
```
<br>
</details>

<details>
<summary>pickle (.pkl) 파일 읽/쓰기</summary>

```py
import pickle

# save
SOMETHING = [1, 2, 3] # example
with open("FILE_NAME.pickle", "wb") as f:
    pickle.dump(SOMETHING, f)

# load
with open("FILE_NAME.pickle", "rb") as f:
    data = pickle.load(f)
```
<br>
</details>


## 🪄 NumPy

`import numpy as np`

<details>
<summary>📦 Load npy, npz file</summary>

```py
# npz: 키 목록 보기
data = np.load(PATH)
keys = [k for k in data.keys()] # print(data.keys())는 안 보임

# 저장된 데이터가 딕셔너리라면
data = (np.load(PATH, allow_pickle=True)).item()
data[KEY] # 키 이용해 데이터 접근
```
<br>
</details>


## 🔥 PyTorch

### Process Image/Video

<details>
<summary>Image at PIL / OpenCV / PyTorch</summary>

| | PIL | OpenCV | PyTorch |
|---|---|---|---|
| load | `Image.open()` | `cv2.imread()` | |
| size func. | `img.size` | `img.shape` | `tensor.shape` or `tensor.size()` |
| size | (w, h) | (h, w, c) | (c, h, w) |
| dtype | 8 (`img.bits`) | uint8 (`img.dtype`) | torch.float32 (0~1) (`tensor.dtype`) |
| range | 0 ~ 255 | 0 ~ 255 | 0 ~ 1 |
| format | RGB (`img.mode`) | BGR | RGB |

- PIL
    ```py
    from PIL import Image
    img = Image.open('path_of_image')

    # PIL -> Numpy
    import numpy as np
    img = np.asarray(img) # or np.array(img)
    # Numpy -> PIL
    img = Image.fromarray(img)
    ```
- OpenCV
    ```py
    import cv2
    img = cv2.imread('path_of_image')
    ```
- PyTorch
    ```py
    # PIL -> tensor
    import torchvision.transforms.functional as F
    img = Image.open('path_of_image')
    img = F.to_tensor(img)
    # Numpy -> tensor
    from torchvision.transforms import ToTensor
    toTensor = ToTensor()
    img = toTensor(img)
    # cv -> tensor (1)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB -> BGR
    img = img.transpose((2, 0, 1)) # H,W,C -> C,H,W
    img = img.float().div(255.0) # normalize
    # cv -> tensor (2)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB -> BGR
    img = img.Tensor(img) # normalize
    img = img.permute(2, 0, 1) # H,W,C -> C,H,W

    # tensor -> PIL, NumPy
    from torchvision.transforms import ToPILImage
    toPILImage = ToPILImage()
    img = toPILImage(img)
    # tensor -> cv
    img = img.detach().cpu().numpy() # tensor -> numpy
    img = np.transpose(img, (1, 2, 0)) # C,H,W -> H,W,C
    img = img*255 # denormalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    img = img.astype(np.uint8).copy() # np.float32 -> np.uint8
    ```

- cv to torch in lambda func.
    - [Reference: Jinsol Kim](https://gaussian37.github.io/dl-pytorch-snippets/#opencv%EB%A1%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A5%BC-%EC%9D%BD%EC%96%B4%EC%84%9C-tensor%EB%A1%9C-%EB%B3%80%ED%99%98-1)
    ```py
    load_images = lambda path, h, w: cv2.resize(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), ((w, h)))
    tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)

    img_tensor = tensorify(load_images("img.png", 400, 300))
    print(img_tensor.shape) # torch.Size([1, 3, 400, 300])
    ```
 
<br>
</details>

<details>
<summary>Save the image</summary>

- Tensor type 
    ```py
    # (1) torchvision
    # Option: nrow (한 줄에 몇 개의 이미지), padding (이미지 간 몇 픽셀 간격), etc
    from torchvision.utils import save_image
    save_iamge(img, 'path_of_image') # (B, C, H, W) -> (W, H, C)

    # (2) plt
    import matplotlib.pyplot as plt
    img = img.permute(1, 2, 0) # [C, H, W] -> [H, W, C]
    ```

- Numpy, PIL type
    ```py
    import numpy as np
    from PIL import Image
    img = Image.fromarray(img) # numpy -> PIL
    img.save('path_of_image', 'jpg')
    ```
</details>

<details>
<summary>Save the video</summary>

```py
import torchvision
# video: np.ndarray, [Time, Hight, Width, Channel], 0~255, np.uint8
torchvision.io.write_video(save_fname, video, fps=fps, audio_codec='aac')
```
</details>


### Use CUDA (GPU)

<details>
<summary>Check CUDA</summary>

```py
import torch; print(torch.cuda.is_available()) # True of False
```
<br>
</details>


<details>
<summary>Setting GPU Devices</summary>

```py
# Method 1)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = <gpu_numbers> 
    # e.g., "1, 2" - assign GPU number 0 and 1 for each GPU 1, GPU 2

# Method 2)
import torch; torch.cuda.set_device(1)

# Method 3) Commandline
CUDA_VISIBLE_DEVICES=2,3 python script_fname.py
```
<br>
</details>



### Checkpoint and Pre-trained model

<details>
<summary>Load model</summary>

- `PATH`: checkpoint file
- `DEVICE`: running device (type: `torch.device`)
- `MODEL`: model to load parameters

```py
checkpoint = torch.load(PATH, map_location=DEVICE)

# if you save only model_state_dict
MODEL.load_state_dict(checkpoint)
# if you save all parameters of model, optimizer, etc.
MODEL.load_state_dict(checkpoint["model_state_dict"])
```
<br>
</details>

### Tensors

<details>
<summary>Merge lists to tensors</summary>

```py
# list[tesor, tensor, ...] -> tensor[tensor, tensor, ...]
torch.stack(list_name, dim=0)
```
<br>
</details>

### Dataset & Dataloader

<details>
<summary>Sampler</summary>

- [커스텀 샘플러 만들기](https://velog.io/@shj4901/PyTorch-Dataset#custom-sampler)
<br>
</details>

## 🚀 Other Python Tools

### wandb

`import wandb`

<details>
<summary>initiate</summary>

```py
# init
wandb.login()
wandb.init(
    project="PROJECT_NAME",
    entity="USER_NAME",
    name="EXPERIMENT_NAME",
    config = {
        "CONFIG1": config1,
    },
    notes="NOTES",
)
```
<br>
</details>

<details>
<summary>Logging</summary>

```py
# logging - number
wandb.log({
    "train/loss1": loss.item(), 
    "val/metric1": metric,
})
# logging - image
# e.g., wandb.log({"result_img": wandb.Image(output_img, mode="RGB", caption="step_2 result")})
wandb.log({'<NAME>': wandb.Image(<IMAGE>, mode="<MODE>", caption="<CAPTION>")})

# logging - video
# e.g., wandb.log({"video": wandb.Video("/home/eungi/video.mp4", fps=30, format="mp4")})
wandb.log({"<NAME>": wandb.Video(<VIDEO_PATH>, fps=<FPS>, format="<FORMAT>")})
```
</details>


### Tensorboard

<details>
<summary>Show tensorboard</summary>

```bash
tensorboard --logdir=<log_directory_path> --port=<port_number>
```
</details>

<details>
<summary>Port forwarding</summary>

```bash
ssh -NfL localhost:<server_port>:localhost:<local_port> <server_name>

# example
ssh -NfL localhost:6007:localhost:6007 eungi@gpu01
```

- References
    - [Remote 서버에서 Tensorboard 연결하기](https://daeun-computer-uneasy.tistory.com/41)
    - [remote server 로부터 Tensorboard 사용하는 방법](https://data-newbie.tistory.com/363)
</details>


## 🔦 Dev Tools

### tmux

<details>
<summary>Usage</summary>

- seesion list: `tmux ls`
- make session: `tmux new -s <session-name>`
- session attach: `tmux a -t <session-name>`
- session detach: `Ctrl + b` → `d`

- Split vertically: `Ctrl + b` → `%`
- Split horizontally: `Ctrl + b` → `>`
- Change focus: `Ctrl + b` → `direction_key` or `space`

- Scroll: `Control + b` → `[` / `q` to quit
</details>

<details>
<summary>Installation</summary>

```bash
### Install

# ubuntu
sudo apt install tmux

# mac
brew install tmux

### Check installation
tmux -V
```
</details>

<details>
<summary>tmux configuration</summary>

Create/Edit `~/.tmux.conf` file:
```bash
vi ~/.tmux.conf
```

the, run:

```bash
tmux source-file ~/.tmux.conf
```

Configs:
- 마우스 사용 허용: `set -g mouse on`
- [Other options](https://velog.io/@suasue/Ubuntu-%ED%84%B0%EB%AF%B8%EB%84%90-%ED%99%94%EB%A9%B4%EB%B6%84%ED%95%A0-Tmux-%EC%89%BD%EA%B2%8C-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)
</details>


### Anaconda

<details>
<summary>Usage</summary>

- environment list: `conda env list`
- create environment: `conda create --name <env_name> [python=<py_version>]`
- remove environment: `conda env remove --name <env_name>`
- clone environment: `conda create --name <AFTER> --clone <ORIGINAL>`
- change environment name: (이름 변경은 지원 안 함) 원하는 이름으로 그 환경을 __복사__해두고 원래 이름의 환경은 지우기
- activate environment: `conda activate <env_name>`
- deactivate environment: `conda deactivate`
- Package List: `conda list`
- Clean: `conda clean --all`
- pip clean: `pip cache purge`

</details>

### Git and GitHub

<details>
<summary>Remote Repository</summary>

```bash
# Check registered remotes
# You can use `git remote get-url origin` instead
git remote -v

# After making an empty repository in GitHub,
# add remote repository in local repository.
git remote add origin <URL-OF-REMOTE-REPOSITORY>

# Push
git branch -M main
git push -u origin master
```
</details>

<details>
<summary>Clone specific vertion of commit</summary>

1. Clone repo: `git clone <repo_address>`
2. Go to that commit: `git reset --hard <commitID>`
</details>

### VSCode

<details>
<summary>Debugging Configurations</summary>

- 항상 특정 파일에서 디버깅하기: `"program": "파일명"` (Note: `${file}`은 디버깅 버튼을 누른 해당 파일을 의미)
- 환경 변수 설정하기: `env` 딕셔너리에 입력. 아래는 GPU 지정 예시.
```
"env": {
	"CUDA_VISIBLE_DEVICES": "6"
}
```
- `python -m`으로 시작하는 실행
```
"module": "dir/.../file" # program  대신
```

</details>

<details>
<summary>Kill VSCode server process</summary>

```bash
# check process list
ps -ef | grep <UserName> | grep vscode
# Kill all processes
#kill -9 $(ps -eL | grep <UserName> | grep vscode)
```
https://bakyeono.net/post/2015-05-05-linux-kill-process-by-name.html
</details>


<details>
<summary>Change all upper/lowercases</summary>

- (Windows) Ctrl + Shift + U
- 변경할 부분 선택 -> Cmd + Shift + P -> `transform to ...`
</details>

<details>
<summary>Extensions</summary>

- indent-rainbow: Colorize indentations
- Comment Anchors: Comment with anchor tags
- Black Formatter: Python code formatter
- isort: Python import part formatter
</details>

<details>
<summary>Rulers (에디터 세로선)</summary>

cmd + shift + P → Open settings (JSON)
```json
"editor.rulers": [
    {
    	"column": 88,
    },
],
```
</details>

### ffmpeg/ffprobe

<details>
<summary>Video <-> image frames</summary>

Options:
- `-ss`/`-to`/`-t`: 추출 시작/종료시점/종료길이 설정. `hh:mm:ss`, `hh:mm:ss.sss`, `s` 형식
- `-framerate`: '입력' 비디오/이미지 스트림의 FPS. 주로 이미지 파일을 비디오로 변환 시 사용
- `-r`: '출력' 파일의 초당 프레임 레이트를 설정. 입력 비디오의 프레임 레이트 조정 혹은 비디오 인코딩 시 사용.
- `-f`: 출력 파일의 포맷 지정. `image2`이면 입력 파일을 비디오가 아니라 이미지로 처리하도록 지시.
- 출력 파일 이름 포맷: `%d`이면 순차적으로 1, 2, 3, ...이고, `%06d`이면 여섯 자리를 맞추되 앞 부분을 0으로 채우는 식.
- `-qscale:v` 또는 `-q:v`: 비디오 품질 비율. 낮을수록 품질 좋고 파일 크기가 큼. 기본 `2`
- `-c:v`: 비디오 코덱 지정
  - `libx264`: H.264 코덱
  - `mpeg4`: MPEG-4 Part 2 코덱. 오래된 장치나 SW의 호환을 위해 사용
  - `copy`: 재인코딩 없이 원본파일에서 그대로 복사(속도 빠름, 품질 손실 없음)
- `pix_fmt`: 비디오의 픽셀 포맷 설정. `yuv420p`이면 H.264에서 널리 사용되는 포맷.

```bash
# Extract frames from a video
ffmpeg -i <VideoPath> -f image2 <ImgPath%d.png>
# ffmpeg -ss 00:01:00 -to 00:21:00 -i input.mp4 -r 25 -f image2 image_%06d.png
# PNG로 변환하지 않으면 화질이 깨질 때가 종종 있음

# Merge frames into single video
ffmpeg -framerate <FPS> -i <PathPattern> -c:v <Value> -pix_fmt <Value> <OutVideoPath.mp4>
# ffmpeg -framerate 25 -i iamge_%03d.png -c:v libx264 -pix_fmt yuv420p <OutVideoPath.mp4>
```
</details>

<details>
<summary>Merge multiple images in one frames</summary>

```bash
# 4 images to one (upper left, upper right, lower left, lower right)
ffmpeg \
-i [ul_path] -i [ur_path] -i [ll_path] -i [lr_path] \
-filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[out]" \
-map", "[out]" \
[save_path]
```

```py
import subprocess
cmd = [
    'ffmpeg', '-y', '-loglevel', "error",
    '-i', ul, '-i', ur, '-i', ll, '-i', lr,
    "-filter_complex",
    "[0:v][1:v]hstack=inputs=2[top];"
    "[2:v][3:v]hstack=inputs=2[bottom];"
    "[top][bottom]vstack=inputs=2[out]",
    "-map", "[out]",
    save_path
]
return_code = subprocess.call(cmd)
```
</details>

<details>
<summary>Extract audio from video</summary>

Options:
- `-ac`: 오디오 채널 설정. `1`은 모노(1채널), `2`는 스테레오(2채널)
- `-vn`: 비디오 스트림 제외
- `-ar`: 오디오 샘플링 레이트 설정
- `-acodec` 혹은 `-c:a`: 오디오 코덱 설정
  - `pcm_s16le`는 비압축 오디오(고품질, 고용량)이며, 확장자는 wav로 저장하는 게 일반적.
  - `aac`는 mp3 대체 위한 고효율 오디오 코덱
  - `copy`이면 별도의 인코딩 없이 원본파일에서 그대로 복사(속도 빠름, 품질 손실 없음)

```bash
ffmpeg -i <VideoPath> -ac 1 -c:a <Value> -ar <SampleRate> -vn <OutPath.[wav/mp4/m4a/aac]>
```
</details>

<details>
<summary>Merge audio and video</summary>

Options:
- `-c:v`: 비디오 코덱 지정
  - `libx264`: H.264 코덱
  - `mpeg4`: MPEG-4 Part 2 코덱. 오래된 장치나 SW의 호환을 위해 사용
  - `copy`: 재인코딩 없이 원본파일에서 그대로 복사(속도 빠름, 품질 손실 없음)
- `-acodec` 혹은 `-c:a`: 오디오 코덱 설정
  - `pcm_s16le`: 비압축 오디오(고품질, 고용량)이며, 확장자는 wav로 저장하는 게 일반적.
  - `aac`: mp3 대체 위한 고효율 오디오 코덱
  - `copy`: 재인코딩 없이 원본파일에서 그대로 복사(속도 빠름, 품질 손실 없음)

```bash
ffmpeg -i <VideoPaht> -i <AudioPath> -c copy -c:v <Value> -c:a <Value> <OutputPath.mp4>
```
</details>

<details>
<summary>Cut the audio</summary>

```bash
ffmpeg -i <AudioPath> -ss <StartTime> -to <EndTime> <OutAudioPath.wav>
```
</details>

<details>
<summary>Give offset (delay) to audio and video</summary>

```bash
ffmpeg -i <VideoPaht> -itsoffset <Offset(sec)> -i <VideoPaht> -map 0:v -map 1:a <OutputPath.mp4>
# map -0:v : 첫 번째 입력 파일을 video 입력으로 삼음
# map -1:a : 두 번째 입력 파일을 audio 입력으로 삼음
```

```py
# 오디오를 뒤로 밀기
subprocess.run(
    f"ffmpeg -loglevel {loglevel} -y "
    + f"-i {video_path} "
    + f"-itsoffset {delay_time} "
    + f"-i {video_path} "
    + "-map 0:v -map 1:a "  # -c:v copy -c:a copy "
    + str(save_path),
    shell=True,
)

# 비디오를 뒤로 밀기
subprocess.run(
    f"ffmpeg -loglevel {loglevel} -y "
    + f"-i {video_path} "
    + f"-itsoffset {delay_time} "
    + f"-i {video_path} "
    + "-map 0:a -map 1:v "  # -c:v copy -c:a copy "
    + str(save_path),
    shell=True,
)
```
</details>

<details>

<summary>ffmpeg: Other Options</summary>

- `-loglevel`: 출력 레벨 설정. /`error`/이면 출력 안 나옴
  - `quiet`: 오류 메시지 외 출력 안 함
  - `panic`, `fatal`: 치명적 오류만 출력
  - `error`: 오류 메시지만 출력
- `-y`: 이미 파일이 있으면 덮어쓰기
- `threads`: 사용할 쓰레드 수 설정. 별도 지정이 없으면 자동으로 최적화.
</details>

<details>
<summary>Information of Video/Audio</summary>

Options:
- `-v`: `error`이면 오류 메시지만 출력하게 해 깔끔한 결과를 제공
- `-show_entries`: 출력할 부분 지정
  - FPS: `stream=r_frame_rate`
  - Duration: `format=duration`
  - Codecs: `stream=codec_type`
- `-of`: 출력 포맷 지정. `json`으로 JSON 형태로 출력 가능.

```bash
# one query. just single line
ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 input.mp4

# multiple query. one line, one output
ffprobe -v error -show_entries format=duration,stream=codec_type -of default=noprint_wrappers=1 input.mp4
```

```py
# If you want to get as scalar value in python pipeline
def get_duration(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    ffprobe_output = json.loads(result.stdout)
    duration = float(ffprobe_output["format"]["duration"])
    return duration
```
</details>

<details>
<summary>How to run in python</summary>

- `subprocess.call(command)`: 명령어를 수행하고 종료 코드를 반환
  - args `command`: 리스트/튜플 혹은 문자열로 전달(`shell=True`일 때만)
  - args `shell`: True일 경우 명렁어를 셸을 통해 실행하고(`command`가 문자영리어야 하며, 파이프나 리디렉션 사용 가능), False(default)일 경우 직적 수행함
  - 명령어 수행의 출력을 받으려면 `subprocess.run()` 사용
  - return이 0이면 성공적으로 수행되었음을 뜻함
	```py
	result = subprocess.call(
	    [
	        "ffmpeg", "-y", "-framerate", "60",
	        "-i", "frame_%04d.png",
	        "-c:v", "libx264",
	        "-pix_fmt", "yuv420p",
	        "output.mp4"
	    ]
	)
	```

- `subprocess.run(command)`: 명령을 실행하고 완료 시까지 대기
  - args `command`: 리스트로 전달
  - `capture_output`: `True` 시 stdout, stderr를 캡쳐함
  - `text`: `True`시 출력을 문자열로 변환함
  ```py
  run(
        f"ffmpeg -loglevel {loglevel} -y "
        + f"-i {video_path} "
        + f"-itsoffset {delay_time} "
        + f"-i {video_path} "
        + "-map 0:a -map 1:v "
        + str(save_path),
        shell=True
  )
  ```

- 만약 터미널에서 잘 작동하는 명령어가 `subprocess`를 했을 때 잘 작동하지 않는다면? (에러, 일부 기능 작동 안 함)
  - ffmpeg의 프로그램 경로를 `ffmpeg` 대신 적어주기
  - `where ffmpeg` → `subprocess.call(["/usr/bin/ffmpeg", ...])`
</details>

### Terminal Customize

<details>
<summary>iTerm2 + oh-my-zsh</summary>

- Change bash to zsh: `chsh -s /usr/bin/zsh`

- https://salmonpack.tistory.com/52
- https://kdohyeon.tistory.com/122
- https://luidy.tistory.com/entry/Terminal-Mac-%ED%84%B0%EB%AF%B8%EB%84%90-%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95%ED%95%98%EA%B8%B0-%EA%BE%B8%EB%AF%B8%EA%B8%B0-iTerm2-oh-my-zsh-tmux
- Key mapping: https://stackoverflow.com/questions/6205157/how-to-set-keyboard-shortcuts-to-jump-to-beginning-end-of-line/29403520#29403520
</details>

### OS 환경변수

- python: `os.environ`으로 설정 (숫자는 str처리)
- 또는 bash 파일에 export로 설정

```
OMP_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_NUM_THREADS

OPENBLAS_NUM_THREADS
VECLIP_MAXIMUM_THREADS
```

## 🐧 Linux

<details>
<summary>Move & Copy Files</summary>

- move file: `mv <from> <to>`
- copy file: `cp <from> <to>`
</details>

<details>
<summary>Count the number of files/directories</summary>

```bash
# All types
ls | wc -l

# Files
ls -l | grep ^- | wc -l

# Directories
ls -l | grep ^d | wc -l
```
</details>

<details>
<summary>Copy file server ↔️ local</summary>

- `scp`

```bash
# if you want to copy directory, add `-r` option
scp -P <PORT_NUM> [OPTIONS] <source> <destination>

# example (server -> local) (run in local)
scp -P PORT_NUM USER@ADDRESS:SERVER_FILE LOCAL_PATH

# example (local -> server) (run in local)
scp -P PORT_NUM LOCAL_FILE USER@ADDRESS:SERVER_PATH
```

- `rsync`
  - `-e 'ssh -p <Port>'`: 포트 변경
  - `-a`: 아카이브 모드. 파일 속성, 심볼릭 링크 등 유지
  - `-v`: 상세 출력
  - `-z`: 전송 중 데이터 압축
  - `-h`: 파일 크기를 사람이 읽기 쉬운 형식으로 표시
  - `--progress`: 전송 진행상황 표시

```bash
# 포트 변경
rsync -avz -e 'ssh -p <Port>' <Src> <Dst>

# 폴더 전체의 진행상황 표시
rsync -avz --info=progress2 <Src> <Dst>
```
</details>

<details>
<summary>Monitor GPU status</summary>

- `nvidia-smi`
    - Keep watching: `watch nvidia-smi`
- `gpustat [OPTIONS]`
    - Install: `pip install gpustat`
    - With `-pi` option, the command runs iteratively
</details>

<details>
<summary>Symbolic Link</summary>

```bash
ln -s [SOURCE] [DEST]

# e.g., you can access 'original.txt' with 'linked.txt'
ln -s /home/eungi/original.txt /home/eungi/yeah/linked.txt

# e.g., you can access 'origin_dir' with 'linked_dir'
# Don't need to make 'linked_dir' first, just type the command blow
# Do not add `/` behind the name of directories
ln -s /home/eungi/origin_dir /home/eungi/linked_dir

# e.g., change link
ln -Tfs [SOURCE] [DEST]
```
</details>

<details>
<summary>Disk usage, File size</summary>

```bash
# `-h` option: print the sizes in human readable format (e.g., 12M)
df -h [PATH] # Disk usage
du -h [--max-depth=0] [PATH] # Size of file/directory
ls -lh [PATH] # just for file
```
</details>

<details>
<summary>CPU/Memory status</summary>

```bash
htop
```
</details>

<details>
<summary>Process</summary>

```bash
# kill
kill -9 PID1 PID2 ...

# process list
ps -e
ps -eL | grep <Query>
```
</details>

<details>
<summary>Find file</summary>

- `find`
	```bash
	find {where-to-find} -name {name} # e.g., find / -name test*
	find {where-to-find} -name {name} -type {type} # e.g., {type} - `d` for directory, `f` for file
	```
- `which`: 실행파일/명령어 위치
- `whereis`: 실행파일, 소스, 매뉴얼 파일 위치 (모든 내용 출력)
</details>

## 🍎 Mac

<details>
<summary>Python & virtual env. in Mac</summary>

- [VSCode에서 파이썬 경로](https://hiddenbeginner.github.io/python/2022/03/16/vscode_terminal_does_not_point_python_of_virtual_envrionment.html)
- [pip 경로](https://velog.io/@csu5216/conda-pip-%EA%B2%BD%EB%A1%9C%EA%B0%80-%EB%8B%A4%EB%A5%B8-%EA%B3%B3%EC%9D%84-%EB%B0%94%EB%9D%BC%EB%B3%BC-%EA%B2%BD%EC%9A%B0-for-MAC)
</details>

## Blender

- [Blender에서 파이썬 사용하기](https://itadventure.tistory.com/319)
