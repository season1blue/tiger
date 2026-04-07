import sys
from .qwen_eval import main


if __name__ == "__main__":
    if "--task-type" not in sys.argv:
        sys.argv.extend(["--task-type", "chair"])
    main()
