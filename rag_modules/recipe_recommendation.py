"""
菜谱推荐模块
负责处理菜谱推荐逻辑和菜谱详情获取
"""

import logging
import json
import random
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RecipeRecommendationManager:
    """
    菜谱推荐管理器

    功能：
    1. 随机菜谱推荐
    2. 菜谱详情获取
    3. 备用推荐数据
    4. 图片URL处理
    """

    def __init__(self, llm_client=None, config=None):
        """初始化推荐管理器"""
        self.index_file = "data/recipes_with_images.json"
        self.dishes_dir = "data/dishes"
        self.llm_client = llm_client
        self.config = config

    def get_random_recipes_with_images(self, limit: int = 3) -> List[Dict[str, Any]]:
        """从预生成的索引文件获取随机的有图片的菜谱推荐"""
        try:
            # 读取预生成的菜谱索引文件
            if not os.path.exists(self.index_file):
                logger.warning(f"菜谱索引文件不存在: {self.index_file}")
                return self._get_fallback_recommendations(limit)

            with open(self.index_file, "r", encoding="utf-8") as f:
                recipes_data = json.load(f)

            if not recipes_data:
                logger.warning("菜谱索引文件为空")
                return self._get_fallback_recommendations(limit)

            # 转换为API格式
            recipes_with_images = []
            for i, recipe_data in enumerate(recipes_data):
                # 生成难度：优先从文件读取，失败则随机
                difficulties = ["easy", "medium", "hard"]
                file_path = recipe_data.get("file_path", "")
                difficulty = self._get_difficulty_from_file(file_path) or random.choice(
                    difficulties
                )

                # 处理图片URL
                image_url = self._process_image_url(recipe_data)

                recipe = {
                    "id": f"recipe_{i + 1}",
                    "name": recipe_data.get("name", "未知菜谱"),
                    "description": recipe_data.get("description", "美味可口的经典菜谱"),
                    "category": recipe_data.get("category", "家常菜"),
                    "imageUrl": image_url
                    or f"https://via.placeholder.com/300x200?text={recipe_data.get('name', 'Recipe')}",
                    "cookingTime": recipe_data.get("cooking_time", 30),
                    "prepTime": 15,
                    "servings": 1,
                    "difficulty": difficulty,
                    "rating": round(random.uniform(4.0, 5.0), 1),  # 随机评分
                    "tags": recipe_data.get("tags", []),
                    "ingredients": [],
                    "steps": [],
                    "markdownPath": recipe_data.get("file_path", ""),
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                }
                recipes_with_images.append(recipe)

            # 随机选择指定数量的菜谱
            if len(recipes_with_images) >= limit:
                selected_recipes = random.sample(recipes_with_images, limit)
            else:
                selected_recipes = recipes_with_images[:limit]
                # 如果不够，用备用数据补充
                if len(selected_recipes) < limit:
                    fallback = self._get_fallback_recommendations(
                        limit - len(selected_recipes)
                    )
                    selected_recipes.extend(fallback)

            logger.info(
                f"从索引文件加载 {len(recipes_with_images)} 个菜谱，返回 {len(selected_recipes)} 个随机推荐"
            )
            return selected_recipes

        except Exception as e:
            logger.error(f"从索引文件获取菜谱失败: {e}")
            return self._get_fallback_recommendations(limit)

    def _get_difficulty_from_file(self, file_path: str) -> Optional[str]:
        """从菜谱文件中读取难度星级，返回 'easy'/'medium'/'hard'，失败返回 None"""
        try:
            if not file_path or not os.path.exists(file_path):
                return None
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "预估烹饪难度" in line:
                        stars = line.count("★")
                        if stars <= 1:
                            return "easy"
                        elif stars <= 3:
                            return "medium"
                        else:
                            return "hard"
            return None
        except Exception as e:
            logger.warning(f"读取难度失败 {file_path}: {e}")
            return None

    def _process_image_url(self, recipe_data: Dict[str, Any]) -> Optional[str]:
        """处理菜谱图片URL"""
        try:
            image_url = recipe_data.get("image_url", "")
            file_path = recipe_data.get("file_path", "")

            if image_url and not image_url.startswith("http"):
                if file_path:
                    # 处理相对路径（去掉 ./ 前缀）
                    if image_url.startswith("./"):
                        image_url = image_url[2:]

                    # 构建GitHub LFS媒体URL
                    # 从file_path中提取dishes目录后的路径
                    if "dishes" in file_path:
                        # 找到dishes的位置，提取dishes后面的路径
                        dishes_index = file_path.find("dishes")
                        if dishes_index != -1:
                            # 提取从dishes开始到文件名之前的路径
                            path_after_dishes = file_path[dishes_index:].replace(
                                "\\", "/"
                            )
                            # 移除文件名，只保留目录路径
                            dir_path = "/".join(path_after_dishes.split("/")[:-1])
                            github_path = f"{dir_path}/{image_url}"
                        else:
                            github_path = image_url
                    else:
                        github_path = image_url

                    # 使用正确的GitHub LFS媒体URL
                    github_base = "https://media.githubusercontent.com/media/FutureUnreal/HowToCook/master/"
                    full_url = github_base + github_path
                    # logger.info(f"转换后的GitHub图片URL: {full_url}")
                    return full_url

            return image_url if image_url.startswith("http") else None

        except Exception as e:
            logger.warning(f"处理图片URL失败: {e}")
            return None

    def _get_fallback_recommendations(self, limit: int = 6) -> List[Dict[str, Any]]:
        """备用推荐菜谱（当数据库查询失败时使用）"""
        fallback_recipes = [
            {
                "id": "fallback_001",
                "name": "红烧肉",
                "description": "肥瘦相间，入口即化的经典家常菜",
                "category": "家常菜",
                "imageUrl": "https://via.placeholder.com/300x200?text=红烧肉",
                "cookingTime": 60,
                "prepTime": 15,
                "servings": 4,
                "difficulty": "medium",
                "rating": 4.8,
                "tags": ["家常菜", "下饭", "经典"],
                "ingredients": [],
                "steps": [],
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            },
            {
                "id": "fallback_002",
                "name": "西红柿鸡蛋",
                "description": "酸甜开胃，营养丰富的国民菜",
                "category": "家常菜",
                "imageUrl": "https://via.placeholder.com/300x200?text=西红柿鸡蛋",
                "cookingTime": 15,
                "prepTime": 10,
                "servings": 2,
                "difficulty": "easy",
                "rating": 4.6,
                "tags": ["简单", "营养", "下饭"],
                "ingredients": [],
                "steps": [],
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            },
            {
                "id": "fallback_003",
                "name": "宫保鸡丁",
                "description": "麻辣鲜香，经典川菜代表",
                "category": "川菜",
                "imageUrl": "https://via.placeholder.com/300x200?text=宫保鸡丁",
                "cookingTime": 25,
                "prepTime": 20,
                "servings": 3,
                "difficulty": "medium",
                "rating": 4.7,
                "tags": ["川菜", "麻辣", "经典"],
                "ingredients": [],
                "steps": [],
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            },
        ]
        return fallback_recipes[:limit]

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取菜谱详情"""
        try:
            # 首先尝试从索引文件获取菜谱信息
            if not os.path.exists(self.index_file):
                logger.warning(f"菜谱索引文件不存在: {self.index_file}")
                return None

            with open(self.index_file, "r", encoding="utf-8") as f:
                recipes_data = json.load(f)

            # 根据ID查找菜谱（ID格式：recipe_1, recipe_2, ...）
            recipe_index = None
            if recipe_id.startswith("recipe_"):
                try:
                    recipe_index = int(recipe_id.split("_")[1]) - 1
                except (ValueError, IndexError):
                    logger.warning(f"无效的菜谱ID格式: {recipe_id}")
                    return None

            if recipe_index is None or recipe_index >= len(recipes_data):
                logger.warning(f"菜谱ID不存在: {recipe_id}")
                return None

            recipe_data = recipes_data[recipe_index]

            # 处理图片URL
            image_url = self._process_image_url(recipe_data)

            # 尝试读取详细内容
            detailed_content = self._read_recipe_markdown(recipe_data.get("name", ""))

            recipe_detail = {
                "id": recipe_id,
                "name": recipe_data.get("name", "未知菜谱"),
                "description": recipe_data.get("description", "美味可口的经典菜谱"),
                "category": recipe_data.get("category", "家常菜"),
                "imageUrl": image_url
                or f"https://via.placeholder.com/300x200?text={recipe_data.get('name', 'Recipe')}",
                "cookingTime": recipe_data.get("cooking_time", 30),
                "prepTime": 15,
                "servings": 2,
                "difficulty": self._get_difficulty_from_file(
                    recipe_data.get("file_path", "")
                )
                or random.choice(["easy", "medium", "hard"]),
                "rating": round(random.uniform(4.0, 5.0), 1),
                "tags": recipe_data.get("tags", []),
                "ingredients": detailed_content.get("ingredients", []),
                "steps": detailed_content.get("steps", []),
                "markdownPath": recipe_data.get("file_path", ""),
                "markdownContent": detailed_content.get("content", ""),
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            }

            logger.info(f"成功获取菜谱详情: {recipe_detail['name']}")
            return recipe_detail

        except Exception as e:
            logger.error(f"获取菜谱详情失败: {e}")
            return None

    def _read_recipe_markdown(self, recipe_name: str) -> Dict[str, Any]:
        """读取菜谱 Markdown 文件，用 LLM 解析食材、步骤、营养信息"""
        try:
            if not os.path.exists(self.dishes_dir):
                logger.warning(f"菜谱目录不存在: {self.dishes_dir}")
                return {"ingredients": [], "steps": [], "content": ""}

            for root, dirs, files in os.walk(self.dishes_dir):
                for file in files:
                    if file.endswith(".md") and recipe_name in file:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        result = self._parse_with_llm(content)
                        if result:
                            result["content"] = content
                            return result

                        logger.warning(f"LLM 解析失败，返回空数据: {recipe_name}")
                        return {"ingredients": [], "steps": [], "content": content}

            logger.warning(f"未找到菜谱文件: {recipe_name}")
            return {"ingredients": [], "steps": [], "content": ""}

        except Exception as e:
            logger.error(f"读取菜谱文件失败: {e}")
            return {"ingredients": [], "steps": [], "content": ""}

    def _parse_with_llm(self, markdown_content: str) -> Optional[Dict[str, Any]]:
        """用 LLM 从 Markdown 中提取食材、步骤、营养信息，返回结构化字典"""
        if not self.llm_client or not self.config:
            logger.warning("llm_client 或 config 未初始化，无法使用 LLM 解析")
            return None

        prompt = f"""请从以下菜谱 Markdown 文本中提取结构化信息，严格按 JSON 格式返回，不要任何其他内容。

菜谱内容：
{markdown_content}

返回格式：
{{
  "ingredients": [
    {{"id": "ing_1", "name": "食材名", "amount": "数量", "unit": "单位", "category": "其他", "isOptional": false}}
  ],
  "steps": [
    {{"id": "step_1", "stepNumber": 1, "title": "步骤 1", "description": "步骤描述", "tips": []}}
  ],
  "nutrition": {{
    "calories": 0,
    "protein": 0,
    "carbs": 0,
    "fat": 0,
    "fiber": 0,
    "sugar": 0,
    "sodium": 0
  }}
}}

规则：
- ingredients 只提取"计算"区域中有明确用量的食材，amount 只填数字（如"200"），unit 填单位（如"g"、"ml"、"个"）
- 如果某食材没有明确数量，amount 填"适量"，unit 填""
- steps 提取"操作"或"步骤"区域的每一个步骤，description 保留完整原文
- nutrition 如果菜谱中没有营养信息，所有字段填 0
- id 字段按 ing_1、ing_2... 和 step_1、step_2... 顺序编号"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()

            # 去掉可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)

            if "ingredients" not in result or "steps" not in result:
                logger.warning("LLM 返回结构不完整")
                return None

            logger.info(
                f"LLM 解析成功：{len(result['ingredients'])} 种食材，{len(result['steps'])} 个步骤"
            )
            return result

        except Exception as e:
            logger.warning(f"LLM 解析失败: {e}")
            return None
