# 第四章：数据层（Data）

## 一、本节学习目标

- 理解项目"零数据库、纯文件"的数据架构思路
- 掌握 JSON / CSV / TXT 三类业务数据的结构
- 能够在工具中按需加载这些数据
- 了解模拟数据与真实数据的对接方式

## 二、核心知识点讲解

### 2.1 为什么用文件而不是数据库？
- **演示项目**：没有真实用户，文件最简单
- **可读性高**：JSON/CSV 直接打开就能看懂
- **无依赖**：不需要 MySQL/Redis 启动服务
- **路径可配置**：在 `config/agent.yml` 中统一管理

### 2.2 三类数据分工
| 数据 | 格式 | 用途 |
|------|------|------|
| 景点信息 | CSV | 景区基础属性 |
| 订单/优惠/路线/讲解 | JSON | 结构化业务数据 |
| 知识库 | TXT/PDF | RAG 检索源 |

### 2.3 配置位置
所有路径都集中在 `config/agent.yml`：
```yaml
external_data_path: data/external/records.csv
tourism_orders_path: data/tourism/orders.json
tourism_scenic_spots_path: data/tourism/scenic_spots.csv
tourism_discounts_path: data/tourism/discounts.json
tourism_routes_path: data/tourism/routes.json
tourism_narrations_path: data/tourism/narrations.json
```

## 三、项目落地场景

- 优惠工具 → 读 `discounts.json` 判定 60-69 / 70+ / 6岁以下 等
- 路线工具 → 读 `routes.json` 取出每段节点、时长、无障碍信息
- 讲解工具 → 读 `narrations.json` 取出点位介绍、亮点、趣味
- 票务客户端 → 读 `scenic_spots.csv` 查基准票价
- 订单工具 → 读 `orders.json` 找历史订单

## 四、关键代码+逐行注释

### 4.1 景点表（CSV）
```csv
spot_id,name,province,city,base_price_adult,base_price_child,base_price_elderly,opening_hours,peak_season_start,peak_season_end,peak_surcharge_percent,has_wheelchair_access,recommended_duration_hours,description
gugong,故宫博物院,北京,北京,60,30,30,08:30-17:00,04-01,10-31,0,true,4,明清两代皇家宫殿 世界最大的宫殿建筑群
badaling,八达岭长城,北京,北京,40,20,20,06:30-19:00,04-01,10-31,0,false,4,明长城精华段落 世界文化遗产
huangshan,黄山风景区,安徽,黄山,190,95,95,06:30-17:00,03-01,11-30,0,false,8,天下第一奇山 世界文化与自然双重遗产
xihu,杭州西湖,浙江,杭州,0,0,0,全天开放,03-01,10-31,0,true,6,中国十大风景名胜之一 世界文化景观遗产
```

### 4.2 优惠规则（JSON）
```json
{
  "elderly": {
    "age_60_69": {
      "discount_type": "半价",
      "discount_rate": 0.5,
      "required_docs": ["身份证"],
      "optional_docs": ["老年证", "老年优待证"],
      "note": "60-69岁老人凭身份证享受半价优惠"
    },
    "age_70_plus": {
      "discount_type": "免票",
      "discount_rate": 1.0,
      "required_docs": ["身份证"],
      "optional_docs": ["老年证"],
      "note": "70岁以上老人凭身份证免票入园"
    }
  },
  "child": {
    "under_6_or_120cm": {
      "discount_type": "免票",
      "discount_rate": 1.0,
      "required_docs": ["户口本", "身份证", "出生证明(可选)"],
      "note": "6岁以下或身高1.2米以下儿童免票"
    }
  }
}
```

### 4.3 路线数据（JSON）
```json
{
  "gugong": {
    "name": "故宫精华游",
    "spot_id": "gugong",
    "duration_hours": 4.0,
    "segments": [
      {"node": "午门", "order": 1, "duration_min": 15, "description": "故宫正门，观赏'凹'字形城门建筑与五门洞", "accessible": true, "rest_facilities": true, "has_narration": true},
      {"node": "太和门", "order": 2, "duration_min": 10, "description": "故宫最大宫门，门前铜狮为全国最大", "accessible": true, "rest_facilities": false, "has_narration": true},
      {"node": "太和殿广场", "order": 3, "duration_min": 25, "description": "太和殿外广场，观赏金銮殿外观、铜龟铜鹤", "accessible": true, "rest_facilities": true, "has_narration": true}
    ],
    "variants": {
      "elderly_accessible": {
        "note": "老人友好路线，减少台阶路段，增加休息点",
        "skip_nodes": [],
        "extra_rest_stops": ["太和殿广场额外休息10分钟"]
      }
    }
  }
}
```

### 4.4 讲解词（JSON）
```json
{
  "gugong": {
    "午门": {
      "intro": "您现在看到的是故宫的正门——午门。午门建于明永乐十八年，也就是1420年，距今已有600多年的历史。午门平面呈'凹'字形，正楼面阔九间，是故宫四座城门中最宏伟的一座。",
      "highlights": ["午门有五个门洞，正中门洞为皇帝专用", "皇后大婚时可走一次中门", "殿试前三名——状元、榜眼、探花可从正门走出一次"],
      "fun_facts": ["民间流传的'推出午门斩首'其实是个误传，明代处决犯人在西四，清代在菜市口", "午门前只举行过'廷杖'——就是打板子的刑罚"],
      "photo_spots": ["午门正前方拍摄全景最佳", "东侧拍摄五凤楼造型"]
    }
  }
}
```

### 4.5 通用加载函数（典型写法）
```python
import json
import csv
from utils.path_tool import get_abs_path
from utils.config_handler import agent_conf
from utils.logger_handler import logger

def load_json_data(key: str) -> dict:
    """根据 agent.yml 中的 key 加载 JSON 数据"""
    try:
        path = get_abs_path(agent_conf[key])
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"[load_json_data]{key} 路径不存在")
        return {}

def load_csv_data(key: str) -> list[dict]:
    """根据 agent.yml 中的 key 加载 CSV 数据（按行返回字典列表）"""
    try:
        path = get_abs_path(agent_conf[key])
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        logger.error(f"[load_csv_data]{key} 路径不存在")
        return []

# 使用示例
discounts = load_json_data("tourism_discounts_path")
scenic_spots = load_csv_data("tourism_scenic_spots_path")
print(discounts["elderly"]["age_60_69"]["discount_type"])   # 半价
print(scenic_spots[0]["name"])                              # 故宫博物院
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] 能正确加载并打印 `discounts.json` 中 `elderly.age_60_69.discount_type`
- [ ] 能遍历 `scenic_spots.csv` 输出所有景区名
- [ ] 修改 `agent.yml` 中某条路径后，工具能正确报"文件不存在"
- [ ] 讲解数据 `narrations.json["gugong"]["午门"]["highlights"]` 返回数组

### 5.2 踩坑避坑点
1. **CSV 第一行是表头**：`csv.DictReader` 自动忽略表头，不要手动 `skip`
2. **JSON 注释不支持**：JSON 标准不允许注释，别写 `//`
3. **中文编码**：所有文件必须 UTF-8，否则 `json.load` 会抛 `UnicodeDecodeError`
4. **路径用配置**：不要在工具函数里硬编码 `data/tourism/...`，必须从 `agent_conf` 取
5. **大数据量注意**：演示项目数据量小直接 `json.load` 即可；真实场景要用流式
