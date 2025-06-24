# utilities/path_config.py
from pathlib import Path
import json
import os

class PathManager:
    def __init__(self, config_name="path_config.json"):
        self.home_dir = Path.home()
        self.project_root = self._find_project_root()
        self.config_file = self.project_root / config_name
        self.paths = self._load_or_create_config()

    def _find_project_root(self):
        """查找项目根目录"""
        current = Path.cwd()
        markers = ['requirements.txt', 'setup.py', 'hace-kv-optimization', 'utilities']

        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent
        return Path.cwd()

    def _load_or_create_config(self):
        """加载或创建路径配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ 已加载现有路径配置: {self.config_file}")
            except Exception as e:
                print(f"⚠️  配置文件损坏，重新创建: {e}")
                config = self._create_default_config()
        else:
            config = self._create_default_config()

        return self._adapt_to_current_user(config)

    def _create_default_config(self):
        """创建默认配置"""
        config = {
            "model_paths": {
                "local_models": str(self.home_dir / "mistral_models"),
                "cache_dir": str(self.home_dir / ".cache" / "huggingface"),
                "default_model": "7B-Instruct-v0.3"
            },
            "data_paths": {
                "datasets_cache": str(self.home_dir / ".cache" / "datasets"),
                "longbench_cache": str(self.home_dir / ".cache" / "longbench"),
                "results_dir": str(self.project_root / "results"),
                "logs_dir": str(self.project_root / "logs")
            },
            "script_mappings": {
                "baseline": "fullkvcache_main.py",
                "baseline_alias": "baseline_main.py",
                "cake": "cake_main.py",
                "h2o": "h2o_main.py"
            },
            "evaluation": {
                "eval_utils_path": str(self.project_root / "eval_utils.py"),
                "baseline_scores_file": str(self.project_root / "baseline_fullkv.json"),
                "enable_scoring": True
            }
        }

        self._save_config(config)
        print(f"✅ 已创建默认路径配置: {self.config_file}")
        return config

    def _adapt_to_current_user(self, config):
        """适配到当前用户目录"""
        adapted = {}
        current_user = self.home_dir.name

        for section, settings in config.items():
            adapted[section] = {}
            for key, path in settings.items():
                if isinstance(path, str) and 'Users' in path:
                    path_obj = Path(path)
                    parts = path_obj.parts
                    user_idx = None
                    for i, part in enumerate(parts):
                        if part == 'Users' and i + 1 < len(parts):
                            user_idx = i + 1
                            break

                    if user_idx:
                        new_parts = parts[:user_idx] + (current_user,) + parts[user_idx+1:]
                        adapted[section][key] = str(Path(*new_parts))
                    else:
                        adapted[section][key] = path
                else:
                    adapted[section][key] = path

        print(f"🔄 已适配路径到用户 {current_user}")
        return adapted

    def _save_config(self, config):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")

    def ensure_directories(self):
        """确保所有目录存在"""
        dirs_to_create = []

        for section, settings in self.paths.items():
            for key, path in settings.items():
                if 'dir' in key or 'path' in key:
                    dirs_to_create.append(Path(path))

        created_count = 0
        for dir_path in dirs_to_create:
            try:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_count += 1
                    print(f"📁 创建目录: {dir_path}")
            except Exception as e:
                print(f"❌ 创建目录失败 {dir_path}: {e}")

        if created_count > 0:
            print(f"✅ 共创建了 {created_count} 个目录")
        else:
            print("ℹ️  所有目录已存在")

    def get_model_path(self):
        """获取模型路径"""
        base_path = self.paths["model_paths"]["local_models"]
        model_name = self.paths["model_paths"]["default_model"]
        return str(Path(base_path) / model_name)

    def get_script_path(self, script_type):
        """获取脚本路径"""
        script_mappings = self.paths["script_mappings"]

        if script_type in script_mappings:
            script_name = script_mappings[script_type]
            script_path = self.project_root / "hace-kv-optimization" / "baselines" / script_name

            if script_path.exists():
                return str(script_path)
            else:
                print(f"⚠️  脚本不存在: {script_path}")
                if script_type == "baseline":
                    alt_script = script_mappings.get("baseline_alias")
                    if alt_script:
                        alt_path = self.project_root / "hace-kv-optimization" / "baselines" / alt_script
                        if alt_path.exists():
                            print(f"🔄 使用备选脚本: {alt_path}")
                            return str(alt_path)

        raise FileNotFoundError(f"未找到 {script_type} 类型的脚本")

    def print_config(self):
        """打印当前配置"""
        print("\n📋 当前路径配置:")
        print("=" * 50)
        for section, settings in self.paths.items():
            print(f"\n[{section}]")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        print("=" * 50)


class EnhancedScriptResolver:
    def __init__(self, base_dir=None):
        self.path_manager = PathManager()
        self.base_dir = Path(base_dir) if base_dir else self.path_manager.project_root / "hace-kv-optimization" / "baselines"

    def get_baseline_script(self):
        """获取正确的基线脚本路径"""
        try:
            return self.path_manager.get_script_path("baseline")
        except FileNotFoundError:
            candidates = ["fullkvcache_main.py", "baseline_main.py", "full_cache_main.py"]

            for candidate in candidates:
                script_path = self.base_dir / candidate
                if script_path.exists():
                    print(f"🎯 找到基线脚本: {script_path}")
                    return str(script_path)

            raise FileNotFoundError("未找到任何基线脚本")

    def validate_script_functionality(self, script_path):
        """验证脚本功能完整性"""
        script_path = Path(script_path)

        if not script_path.exists():
            return False, "脚本文件不存在"

        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            required_features = ["def main(", "scoring", "baseline", "experiment"]
            missing_features = []
            for feature in required_features:
                if feature not in content:
                    missing_features.append(feature)

            if missing_features:
                return False, f"缺少功能: {missing_features}"

            return True, "脚本功能完整"

        except Exception as e:
            return False, f"脚本验证失败: {e}"
