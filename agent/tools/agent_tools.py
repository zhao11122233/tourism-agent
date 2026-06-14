import json
import os
import re
import random
import datetime
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from client.ticket_client import get_ticket_client, TicketClientError

rag = RagSummarizeService()

# ============================================================
# 数据层 — 景区价格从 data/pricing/scenic_spots.json 加载，优惠规则/路线/讲解
# ============================================================

_pricing_data = None

def _get_pricing():
    """加载景区价格统一数据（模块级缓存，agent_tools 和 ticket_client 共享数据源）"""
    global _pricing_data
    if _pricing_data is None:
        _pricing_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'pricing', 'scenic_spots.json')
        with open(_pricing_path, 'r', encoding='utf-8') as _f:
            _pricing_data = json.load(_f)
    return _pricing_data

def _in_date_range(month_day: str, start: str, end: str) -> bool:
    """判断 MM-DD 是否在 [start, end] 区间内（支持跨年区间如 11-01 ~ 03-31）"""
    if start <= end:
        return start <= month_day <= end
    else:
        return month_day >= start or month_day <= end

def _resolve_spot(name: str):
    """根据景区名或别名查找景区价格数据，返回 (canonical_name, spot_data) 或 (None, None)"""
    data = _get_pricing()
    spots = data["scenic_spots"]
    if name in spots:
        return name, spots[name]
    for cname, spot in spots.items():
        if name in cname or cname in name:
            return cname, spot
        if name in spot.get("aliases", []):
            return cname, spot
    for cname, spot in spots.items():
        for alias in spot.get("aliases", []):
            if alias in name or name in alias:
                return cname, spot
    return None, None

def _get_platform_fees():
    return _get_pricing()["platform_fees"]

PERSON_TYPE_DISCOUNTS = {
    "普通成人": {"discount_type": "全价", "discount_rate": 0.0, "required_docs": ["身份证"]},
    "成人": {"discount_type": "全价", "discount_rate": 0.0, "required_docs": ["身份证"]},
    "儿童_6岁以下": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["户口本", "身份证(可选)"]},
    "儿童_6至18岁": {"discount_type": "半价", "discount_rate": 0.5, "required_docs": ["户口本", "学生证(可选)"]},
    "老人_60至69岁": {"discount_type": "半价", "discount_rate": 0.5, "required_docs": ["身份证"]},
    "老人_70岁以上": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["身份证"]},
    "军人_现役": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["军官证或士兵证"]},
    "军人_退伍": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["退伍军人优待证"]},
    "残疾人_1至2级": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["残疾人证(1-2级)"], "companion_discount": 0.5},
    "残疾人_3至4级": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["残疾人证(3-4级)"]},
    "聋哑人士": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["残疾人证(听力/语言类别)"]},
    "学生": {"discount_type": "半价", "discount_rate": 0.5, "required_docs": ["学生证", "身份证"]},
}

ROUTE_TEMPLATES = {
    "故宫": {
        "name": "故宫精华游",
        "duration_hours": 4.0,
        "segments": [
            {"node": "午门", "duration_min": 15, "desc": "故宫正门，五门洞与'凹'字形城楼", "accessible": True, "rest": True},
            {"node": "太和殿广场", "duration_min": 25, "desc": "金銮殿外观、铜龟铜鹤", "accessible": True, "rest": True},
            {"node": "中和殿/保和殿", "duration_min": 20, "desc": "皇帝休息与殿试场所", "accessible": True, "rest": False},
            {"node": "乾清宫", "duration_min": 20, "desc": "皇帝寝宫，'正大光明'匾", "accessible": True, "rest": True},
            {"node": "御花园", "duration_min": 25, "desc": "皇家园林，连理柏与堆秀山", "accessible": True, "rest": True},
        ],
        "elderly_note": "全程无障碍，建议放慢节奏至正常速度的60%",
        "wheelchair_note": "午门西侧有专用无障碍通道，各殿有电梯"
    },
    "八达岭长城": {
        "name": "八达岭长城精华线",
        "duration_hours": 4.0,
        "segments": [
            {"node": "登城口", "duration_min": 10, "desc": "景区入口，照相留念", "accessible": True, "rest": True},
            {"node": "北一楼至北四楼", "duration_min": 50, "desc": "好汉坡与好汉碑打卡", "accessible": False, "rest": True},
            {"node": "北五楼至北八楼", "duration_min": 60, "desc": "最高点北八楼，海拔1015米", "accessible": False, "rest": True},
        ],
        "elderly_note": "建议只走到北四楼，乘坐缆车上下",
        "wheelchair_note": "仅限登城口平台及缆车上站观景台"
    },
    "黄山": {
        "name": "黄山经典一日游",
        "duration_hours": 8.0,
        "segments": [
            {"node": "云谷索道上山", "duration_min": 20, "desc": "缆车俯瞰竹海云海", "accessible": False, "rest": True},
            {"node": "始信峰", "duration_min": 30, "desc": "黄山松景观：黑虎松、连理松", "accessible": False, "rest": False},
            {"node": "北海景区", "duration_min": 40, "desc": "清凉台、猴子观海、梦笔生花", "accessible": False, "rest": True},
            {"node": "光明顶", "duration_min": 40, "desc": "第二高峰1860米，360度全景", "accessible": False, "rest": True},
            {"node": "迎客松/玉屏楼", "duration_min": 30, "desc": "黄山标志迎客松，树龄约1300年", "accessible": False, "rest": True},
        ],
        "elderly_note": "建议跳过光明顶陡峭路段，增加休息时间",
        "wheelchair_note": "仅限索道上站和北海景区部分路段"
    },
    "秦始皇兵马俑博物馆": {
        "name": "兵马俑博物馆精华游",
        "duration_hours": 3.0,
        "segments": [
            {"node": "一号坑展厅", "duration_min": 50, "desc": "最大展厅，约6000件陶俑陶马", "accessible": True, "rest": True},
            {"node": "二号坑展厅", "duration_min": 25, "desc": "多兵种混合军阵", "accessible": True, "rest": False},
            {"node": "三号坑展厅", "duration_min": 15, "desc": "军阵指挥部", "accessible": True, "rest": False},
            {"node": "铜车马展厅", "duration_min": 25, "desc": "两乘精美青铜车马", "accessible": True, "rest": True},
        ],
        "elderly_note": "全馆室内无障碍，建议放慢节奏",
        "wheelchair_note": "全馆无障碍设施完善，可到达所有展厅"
    },
}

# 点位→所属景区映射表（用于细粒度定位和上下文关联）
POINT_TO_SPOT = {
    # 故宫
    "午门": "故宫", "太和殿": "故宫", "太和门": "故宫", "中和殿": "故宫", "保和殿": "故宫",
    "乾清宫": "故宫", "乾清门": "故宫", "交泰殿": "故宫", "坤宁宫": "故宫",
    "御花园": "故宫", "神武门": "故宫", "东华门": "故宫", "西华门": "故宫",
    "文华殿": "故宫", "武英殿": "故宫", "养心殿": "故宫", "慈宁宫": "故宫",
    # 八达岭长城
    "北一楼": "八达岭长城", "北二楼": "八达岭长城", "北三楼": "八达岭长城", "北四楼": "八达岭长城",
    "北五楼": "八达岭长城", "北六楼": "八达岭长城", "北七楼": "八达岭长城", "北八楼": "八达岭长城",
    "北九楼": "八达岭长城", "北十楼": "八达岭长城", "北十一楼": "八达岭长城", "北十二楼": "八达岭长城",
    "南一楼": "八达岭长城", "南二楼": "八达岭长城", "南三楼": "八达岭长城", "南四楼": "八达岭长城",
    "南五楼": "八达岭长城", "南六楼": "八达岭长城", "南七楼": "八达岭长城",
    "好汉坡": "八达岭长城", "登城口": "八达岭长城",
    # 慕田峪长城
    "大角楼": "慕田峪长城", "正关台": "慕田峪长城", "慕田峪北一楼": "慕田峪长城",
    # 黄山
    "迎客松": "黄山", "始信峰": "黄山", "光明顶": "黄山", "莲花峰": "黄山", "天都峰": "黄山",
    "西海大峡谷": "黄山", "北海景区": "黄山", "玉屏楼": "黄山", "飞来石": "黄山",
    "云谷寺": "黄山", "排云亭": "黄山", "猴子观海": "黄山", "梦笔生花": "黄山",
    # 西湖
    "雷峰塔": "西湖", "断桥": "西湖", "苏堤": "西湖", "三潭印月": "西湖", "岳王庙": "西湖", "灵隐寺": "西湖",
    # 兵马俑
    "一号坑": "秦始皇兵马俑博物馆", "二号坑": "秦始皇兵马俑博物馆", "三号坑": "秦始皇兵马俑博物馆",
    "铜车马": "秦始皇兵马俑博物馆", "文物陈列厅": "秦始皇兵马俑博物馆",
    # 颐和园
    "万寿山": "颐和园", "佛香阁": "颐和园", "长廊": "颐和园", "十七孔桥": "颐和园", "昆明湖": "颐和园",
    # 张家界
    "袁家界": "张家界", "天子山": "张家界", "金鞭溪": "张家界", "乾坤柱": "张家界",
    # 九寨沟
    "五花海": "九寨沟", "珍珠滩瀑布": "九寨沟", "诺日朗瀑布": "九寨沟",
    # 布达拉宫
    "红宫": "布达拉宫", "白宫": "布达拉宫",
    # 桂林
    "九马画山": "桂林漓江", "黄布倒影": "桂林漓江",
}

NARRATION_MATERIALS = {
    "故宫": {
        "午门": "午门是故宫的正门，建于明永乐十八年（1420年）。午门平面呈'凹'字形，有五个门洞——正中门洞为皇帝专用，皇后大婚时可走一次，殿试前三名（状元、榜眼、探花）可从此门走出一次。民间流传的'推出午门斩首'其实是个误传，明清两代处决犯人都在菜市口等地。",
        "太和殿广场": "您面前就是故宫最大的宫殿——太和殿，俗称'金銮殿'。它建在三层汉白玉台基之上，面阔十一间，是中国现存最大的木结构宫殿。殿前铜龟、铜鹤各一对，象征江山永固、万寿无疆。太和殿主要用于皇帝登基、大婚等重大典礼。",
        "乾清宫": "乾清宫在明代和清前期是皇帝的寝宫。宫内正中悬挂'正大光明'匾额，为顺治皇帝御笔。雍正皇帝创建秘密立储制度后，传位诏书就藏在此匾之后，这是清代皇位继承制度的一大创举。",
        "御花园": "御花园位于故宫中轴线最北端，是皇帝和后妃们休憩的地方。园内有古树160余棵，其中最著名的是连理柏——两棵柏树自然交缠生长在一起。千秋亭和万春亭的藻井极为精美，是不可错过的看点。",
    },
    "八达岭长城": {
        "登城口": "欢迎来到八达岭长城！这里是明长城的精华段落，由抗倭名将戚继光督建。长城墙体平均高7.8米，可五马并驰。八达岭地势险要，自古有'居庸之险不在关而在八达岭'之说。",
        "北一楼": "您现在位于八达岭长城北段的第一座敌楼——北一楼。这里是北段长城的起点，敌楼为两层砖石结构，上层设有瞭望口和射孔。从这里开始，长城蜿蜒而上，前方就是著名的好汉坡。北一楼地势相对平缓，是拍照留念的好位置。",
        "北四楼": "北四楼是八达岭北段长城的重要节点，海拔约888米，这里设有著名的'好汉碑'。'不到长城非好汉'的石刻就在这里，是游客必到的打卡点。从此处眺望，长城如巨龙蜿蜒于燕山山脉之中，景色壮丽。",
        "北八楼": "恭喜您到达八达岭长城最高点——北八楼，海拔约1015米！这里是八达岭长城海拔最高的敌楼，视野极为开阔，可360度俯瞰群山。站在这里，您能真正体会到长城的雄伟壮观和古代军事防御工程的智慧。",
        "好汉坡": "前方就是著名的好汉坡，坡度约45度，是八达岭最陡峭的段落，共约200级台阶。'不到长城非好汉'——爬上去，您就是好汉！爬的时候注意安全，扶好扶手，量力而行。",
    },
    "慕田峪长城": {
        "登城口": "欢迎来到慕田峪长城！慕田峪长城位于北京市怀柔区，是明代长城保存最完好的段落之一。与八达岭相比，慕田峪植被覆盖率高达96%，有'万里长城，慕田峪独秀'的美誉。这里游客相对较少，非常适合深度游览。",
        "北一楼": "您现在位于慕田峪长城北段第一座敌楼。慕田峪的敌楼独具特色——每座敌楼都有不同的内部结构，有的是空心回廊式，有的是中心室式。从这里向北望去，长城如游龙般穿行于青山翠谷之间，景色非常壮观。",
        "大角楼": "大角楼是慕田峪长城最具特色的敌楼之一，因三面有长城交汇而得名。这里是长城少有的'三岔口'——一条向西去往北京方向，一条向东通往古北口，一条向北延伸。站在楼上，可以看到长城在三个方向上延伸的独特景观。",
    },
    "黄山": {
        "始信峰": "俗话说'不到始信峰，不见黄山松'。您可以看到著名的黑虎松——树身高大、枝叶浓密，远望如黑色猛虎盘踞山间，树龄约450年。还有连理松，一株双干并蒂而生，象征永恒的爱情。",
        "迎客松": "这就是闻名天下的黄山迎客松！树龄约1300年。看那一侧树枝向前伸出，像一个人伸出手臂欢迎远方的客人。迎客松是中国的文化符号，黄山对它实行24小时专人看护——这是全国唯一有'警卫'的树。",
    },
    "秦始皇兵马俑博物馆": {
        "一号坑": "您现在进入的是一号坑——兵马俑中规模最大的坑，面积约14260平方米。坑内约有6000件陶俑和陶马，按实战军阵排列。最震撼的是——每一个陶俑的面部表情都各不相同，千人千面，无一雷同。",
        "铜车马": "这是秦始皇陵出土的铜车马，每乘车由3000多个零部件组成，运用了铸造、焊接、镶嵌等多种工艺。铜车马的伞盖最薄处仅1毫米，可以灵活开合，展现了2000多年前登峰造极的冶金技术。",
    },
}

SMS_FORMATS = {
    "meituan": {"length": 6, "charset": "纯数字", "template": "6位数字", "validity_min": 5, "sender": "106900****"},
    "ctrip": {"length": 8, "charset": "大写字母+数字混合", "template": "8位字母数字", "validity_min": 10, "sender": "106980****"},
    "spot_self": {"length": "4-6", "charset": "纯数字", "template": "4-6位数字", "validity_min": 15, "sender": "景区服务号"},
}

PLATFORM_ORDER_SCHEMAS = {
    "meituan": {"order_prefix": "MT", "required_fields": ["order_id", "scenic_spot", "ticket_type", "quantity", "total_price", "visitor_name", "id_card", "phone", "sms_code"]},
    "ctrip": {"order_prefix": "CTRIP", "required_fields": ["order_id", "scenic_spot", "ticket_type", "quantity", "total_price", "visitor_name", "id_card", "phone", "sms_code"]},
    "spot_self": {"order_prefix": "SPOT", "required_fields": ["order_id", "scenic_spot", "ticket_type", "quantity", "total_price", "visitor_name", "id_card", "phone", "id_card_verified"]},
}



# ============================================================
# 8 个自定义工具
# ============================================================

@tool
def parse_user_info(query: str) -> str:
    """解析用户原始查询，提取出行人数、人员类型（老人/儿童/军人/聋哑/残障/普通成人）和核心诉求（购票/政策查询/路线规划/导览/凭证核验），返回结构化JSON字符串。

    入参 query 为用户原始输入的自然语言字符串。
    """
    result = {
        "traveler_count": 1,
        "person_types": [],
        "core_need": "general",
        "raw_query": query,
        "extracted_ages": [],  # 提取到的具体年龄如 ["5岁","59岁","56岁"]
    }

    # 提取具体年龄信息
    age_set = set()
    # 模式1: "X岁"（如"5岁""59岁""老人70岁"）
    for m in re.finditer(r'(\d+)\s*岁', query):
        age_set.add(f"{m.group(1)}岁")
    # 模式2: 人员词后的孤立数字（如"大人59""老人70"）
    for m in re.finditer(r'(?:大人|老人|成人|小孩|儿童|孩子)\s*(\d{1,3})', query):
        num = int(m.group(1))
        if num >= 3:  # 排除数量词（1-2可能是数量）
            age_set.add(f"{num}岁")
    result["extracted_ages"] = sorted(age_set, key=lambda x: int(re.search(r'\d+', x).group()))

    # 人数提取：汇总所有匹配到的数字前缀（如"2个老人""1个儿童" → 2+1=3）
    count_patterns = [
        r'(\d+)\s*个', r'(\d+)\s*位', r'(\d+)\s*名',
        r'([一二两三四五六七八九十])\s*个', r'([一二两三四五六七八九十])\s*位',
    ]
    num_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    total_count = 0
    for pat in count_patterns:
        for m in re.finditer(pat, query):
            val = m.group(1)
            if val.isdigit():
                n = int(val)
            else:
                n = num_map.get(val)
                if n is None:
                    n = int(val)
            total_count += n
    if total_count > 0:
        result["traveler_count"] = total_count

    # 人员类型识别
    if any(w in query for w in ["老人", "老年", "爸妈", "父母", "爷爷奶奶", "外公外婆", "60岁", "70岁"]):
        result["person_types"].append("老人")
    if any(w in query for w in ["儿童", "小孩", "孩子", "宝宝", "小朋友", "6岁", "8岁", "10岁", "12岁"]):
        result["person_types"].append("儿童")
    if any(w in query for w in ["军人", "军官", "士兵", "退伍", "退役", "武警", "消防"]):
        result["person_types"].append("军人")
    if any(w in query for w in ["聋哑", "听力障碍", "语言障碍", "听障"]):
        result["person_types"].append("聋哑人士")
    if any(w in query for w in ["残疾", "残障", "残疾人", "残疾证"]):
        result["person_types"].append("残疾人")
    if any(w in query for w in ["成人", "大人", "普通", "全价", "成年人"]):
        result["person_types"].append("普通成人")
    if any(w in query for w in ["学生", "大学生", "中小学生"]):
        result["person_types"].append("学生")

    # 如果没有识别到特殊人群，默认为普通成人
    if not result["person_types"]:
        result["person_types"].append("普通成人")

    # 模式3: 空格/逗号分隔的2-3位数字（如"59 56"），可能为年龄
    if result["person_types"] and len(age_set) > 0:
        bare_nums = re.findall(r'(?<!\d)(\d{2,3})(?!\s*[个位名岁月日])', query)
        for n in bare_nums:
            num = int(n)
            if 3 <= num <= 120:
                age_set.add(f"{num}岁")
    result["extracted_ages"] = sorted(age_set, key=lambda x: int(re.search(r'\d+', x).group()))

    # 根据年龄细化人员类型，使 verify_discount 能精确匹配年龄档位
    if result["extracted_ages"]:
        age_values = []
        for a in result["extracted_ages"]:
            m = re.search(r'\d+', a)
            if m:
                age_values.append(int(m.group()))

        refined = []
        remaining = list(age_values)

        for ptype in result["person_types"]:
            if ptype == "儿童":
                matched = [a for a in remaining if a < 18]
                for age in matched:
                    refined.append("儿童_6岁以下" if age < 6 else "儿童_6至18岁")
                    remaining.remove(age)
                if not matched:
                    refined.append(ptype)
            elif ptype == "老人":
                matched = [a for a in remaining if a >= 60]
                for age in matched:
                    refined.append("老人_70岁以上" if age >= 70 else "老人_60至69岁")
                    remaining.remove(age)
                if not matched:
                    refined.append(ptype)
            else:
                refined.append(ptype)

        result["person_types"] = refined

    # 核心诉求识别
    if any(w in query for w in ["订票", "预订", "帮我订", "下单", "出单", "预订单", "生成订单", "确认订单"]):
        result["core_need"] = "order_generation"
    elif any(w in query for w in ["查票", "余票", "有没有票", "还有票吗", "剩多少", "还有多少"]):
        result["core_need"] = "ticketing"
    elif any(w in query for w in ["买票", "购票"]):
        result["core_need"] = "ticketing"
    elif any(w in query for w in ["票价", "价格", "多少钱", "费用", "打折", "能不能免", "免门票", "免票吗", "免费吗", "要门票吗", "要不要钱"]):
        result["core_need"] = "policy_inquiry"
    elif any(w in query for w in ["路线", "规划", "行程", "路线图", "怎么走", "游玩路线", "一日游", "半日游"]):
        result["core_need"] = "route_planning"
    elif any(w in query for w in ["讲解", "导览", "介绍", "解说", "历史", "典故", "好玩"]):
        result["core_need"] = "narration"
    elif any(w in query for w in ["核验", "入园", "凭证", "验证", "订单查询", "检票", "二维码", "短信"]):
        result["core_need"] = "credential_verification"
    elif any(w in query for w in ["政策", "优惠规定", "免费政策", "半价", "免票", "残疾证", "军官证", "老人证", "有什么优惠"]):
        result["core_need"] = "policy_inquiry"
    else:
        # 隐式意图识别：无明确关键词但有人口+景区信息 → 默认定价咨询
        has_people_info = bool(result["person_types"] != ["普通成人"] or result["extracted_ages"] or result["traveler_count"] > 1)
        has_scenic = any(w in query for w in ["故宫", "长城", "黄山", "西湖", "兵马俑", "颐和园", "张家界", "九寨沟", "布达拉宫", "漓江"])
        if has_people_info and has_scenic:
            result["core_need"] = "policy_inquiry"
        else:
            result["core_need"] = "general_inquiry"

    # 检测缺失的关键信息，直接内嵌追问提示（减少后续工具调用，防止循环）
    result["needs_info"] = False
    result["missing_info_prompt"] = ""
    missing_items = []

    # 政策咨询场景：必须知道年龄+景区
    if result["core_need"] in ("policy_inquiry", "ticketing"):
        has_child = any("儿童" in t for t in result["person_types"])
        has_elderly = any("老人" in t for t in result["person_types"])
        if has_child and not result["extracted_ages"]:
            result["needs_info"] = True
            missing_items.append("孩子的具体年龄或身高（6岁以下/1.2米以下免票，6-18岁半价）")
        if has_elderly and not result["extracted_ages"]:
            result["needs_info"] = True
            missing_items.append("老人的具体年龄（60-69岁半价，70岁以上免票）")
        if not any(w in query for w in ["故宫", "长城", "黄山", "西湖", "兵马俑", "颐和园", "张家界", "九寨沟", "布达拉宫", "漓江", "景区", "景点"]):
            result["needs_info"] = True
            missing_items.append("计划前往的景区名称")
        # 政策咨询场景绝对不能索要身份证/手机/平台

    if result["needs_info"]:
        items_str = "、".join(f"{i+1}.{item}" for i, item in enumerate(missing_items))
        result["missing_info_prompt"] = (
            f"想要为您准确判断，还需要补充以下信息：{items_str}\n"
            f"请一次性提供，我会立即为您查询。"
        )

    logger.info(f"[parse_user_info]解析结果：{json.dumps(result, ensure_ascii=False)}")
    return json.dumps(result, ensure_ascii=False)


@tool
def search_policy(query: str) -> str:
    """从向量知识库检索文旅票务规则、特殊人群优惠政策、入园凭证核验规则、景区导览材料、多平台订单数据结构和短信验证码格式等专业资料，返回检索到的知识内容字符串。

    入参 query 为检索关键词或问题描述字符串。
    """
    return rag.rag_summarize(query)


@tool
def verify_discount(person_types: str, certificates: str = "{}") -> str:
    """校验人员类型的优惠资格。对照内置优惠规则和知识库政策，逐一判定每类人员的优惠等级（免票/半价/全价），列出所需证件和注意事项，返回JSON字符串。

    入参 person_types 为JSON数组字符串如'["老人_70岁以上","儿童_8岁","普通成人"]'，
    certificates 为JSON对象字符串如'{"身份证":true,"残疾证":false,"军官证":false}'，可选，默认为空对象。
    """
    # 容错1: 检测 LLM 错误地将两个参数打包成一层 JSON 的情况
    if isinstance(person_types, str):
        pt = person_types.strip()
        # 多次尝试解析：原始 → 剥外层引号 → 剥外层大括号
        for i in range(3):
            try:
                parsed = json.loads(pt)
                if isinstance(parsed, dict):
                    if "person_types" in parsed:
                        person_types = parsed["person_types"]
                    elif len(parsed) == 1:
                        # 取第一个值作为 person_types
                        person_types = list(parsed.values())[0]
                    if "certificates" in parsed:
                        certificates = parsed["certificates"]
                    break
                elif isinstance(parsed, list):
                    person_types = pt  # 已经是一个有效的数组字符串
                    break
            except (json.JSONDecodeError, TypeError):
                pt = pt.strip("'\"")  # 剥一层再试
                if i == 2:
                    pass  # 三次都失败，使用原始值

    # 容错2: 多种解析尝试
    if isinstance(person_types, str):
        pt_stripped = person_types.strip().strip("'\"")  # 去掉可能的外层引号
        for attempt in [person_types, pt_stripped]:
            try:
                types_list = json.loads(attempt)
                if isinstance(types_list, list):
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            types_list = [person_types]
    else:
        types_list = person_types
    if certificates is None:
        certificates = "{}"
    try:
        certs = json.loads(certificates) if isinstance(certificates, str) else certificates
    except json.JSONDecodeError:
        certs = {}

    results = []
    for ptype in types_list:
        ptype_clean = ptype.strip()
        # 尝试精确匹配 -> 模糊匹配
        matched = None
        for key, rule in PERSON_TYPE_DISCOUNTS.items():
            # 模糊匹配：如果 person_types 包含 key 的关键词 或 key 包含 person_types 的关键词
            if ptype_clean in key or key in ptype_clean:
                matched = rule
                break
            # 更宽松的匹配：提取关键词
            keywords_map = {
                "老人": "老人", "儿童": "儿童", "军人": "军人", "残疾": "残疾人",
                "聋哑": "聋哑人士", "学生": "学生", "成人": "普通成人", "普通": "普通成人",
            }
            for kw, mapped in keywords_map.items():
                if kw in ptype_clean and mapped in key:
                    matched = rule
                    break
            if matched:
                break

        if matched:
            entry = {
                "person_type": ptype_clean,
                "discount_type": matched["discount_type"],
                "discount_rate": matched["discount_rate"],
                "required_docs": matched["required_docs"],
                "companion_discount": matched.get("companion_discount", 0.0),
                "verified": True,
                "notes": matched.get("discount_type", "") + "优惠已确认",
            }
        else:
            entry = {
                "person_type": ptype_clean,
                "discount_type": "全价",
                "discount_rate": 0.0,
                "required_docs": ["身份证"],
                "companion_discount": 0.0,
                "verified": False,
                "notes": "未匹配到特定优惠规则，按全价处理，请确认人员类型",
            }

        # 检查是否有对应证件
        required = entry["required_docs"]
        missing = [d for d in required if not certs.get(d, False) and not any(
            kw in str(certs).lower() for kw in d.lower().split("或") if kw
        )]
        if missing:
            entry["missing_documents"] = missing
            entry["notes"] += f"；需补充证件：{'/'.join(missing)}"

        results.append(entry)

    logger.info(f"[verify_discount]判定结果：{json.dumps(results, ensure_ascii=False)}")
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def calc_ticket_price(scenic_spot: str, person_details: str = '[{"person_type":"普通成人","discount_type":"全价","discount_rate":0.0}]', platform: str = "meituan", visit_date: str = "") -> str:
    """根据景区名称、已验证优惠的人员详情、购票平台和游玩日期，自动查询基准票价（含淡旺季）、应用折扣和平台附加费，计算最优总价，返回详细费用明细JSON字符串。

    入参 scenic_spot 为景区名称，person_details 为 verify_discount 返回的JSON字符串（可选，默认按普通成人全价计算），
    platform 为购票平台标识（meituan/ctrip/spot_self），可选，默认为meituan，
    visit_date 为游玩日期(YYYY-MM-DD格式)，用于判断淡旺季，可选，默认按旺季计算。
    """
    if person_details is None:
        person_details = '[{"person_type":"普通成人","discount_type":"全价","discount_rate":0.0}]'
    try:
        details = json.loads(person_details) if isinstance(person_details, str) else person_details
    except json.JSONDecodeError:
        details = [{"person_type": "普通成人", "discount_type": "全价", "discount_rate": 0.0}]

    # 查找景区价格
    cname, spot_data = _resolve_spot(scenic_spot.strip())

    if not spot_data:
        return json.dumps({
            "error": f"未找到景区'{scenic_spot}'的价格信息",
            "available_spots": list(_get_pricing()["scenic_spots"].keys()),
        }, ensure_ascii=False)

    # 判断淡旺季
    if visit_date and len(visit_date) >= 10:
        month_day = visit_date[5:10]  # "MM-DD"
        peak = spot_data["peak_season"]
        if _in_date_range(month_day, peak["start"], peak["end"]):
            season = "peak"
        else:
            season = "off_peak"
    else:
        season = "peak"  # 未提供日期默认旺季

    season_name = spot_data[f"{'peak' if season == 'peak' else 'off_peak'}_season"]["name"]
    prices = spot_data[f"prices_{season}"]

    # 平台费用
    platform_fees = _get_platform_fees()
    pf = platform_fees.get(platform, platform_fees["spot_self"])

    breakdown = []
    total = 0.0
    total_discount = 0.0

    for person in details:
        ptype = person.get("person_type", "普通成人")
        discount_rate = person.get("discount_rate", 0.0)

        # 确定基准票价：按人员类型匹配 JSON 中的价格键
        if "老人" in ptype:
            base_price = prices["elderly"]
        elif "儿童" in ptype:
            base_price = prices["child"]
        elif "学生" in ptype:
            base_price = prices["student"]
        else:
            base_price = prices["adult"]

        # 应用优惠折扣
        discount_amount = round(base_price * discount_rate, 2)
        final_price = round(base_price - discount_amount, 2)

        entry = {
            "person_type": ptype,
            "base_price": base_price,
            "discount_rate": discount_rate,
            "discount_amount": discount_amount,
            "discount_reason": person.get("discount_type", ""),
            "final_price": final_price,
            "season": season_name,
        }

        # 陪护优惠
        companion_discount = person.get("companion_discount", 0.0)
        if companion_discount > 0:
            companion_price = round(base_price * (1 - companion_discount), 2)
            entry["companion_price"] = companion_price
            entry["companion_discount"] = companion_discount

        breakdown.append(entry)
        total += final_price
        total_discount += discount_amount

    # 平台服务费
    service_fee = pf.get("service_fee", 0.0)
    total_with_fee = round(total + service_fee, 2)

    result = {
        "scenic_spot": cname,
        "platform": platform,
        "season": season_name,
        "base_total": round(sum(p["base_price"] for p in breakdown), 2),
        "total_discount": round(total_discount, 2),
        "subtotal": round(total, 2),
        "platform_service_fee": service_fee,
        "total_price": total_with_fee,
        "price_breakdown": breakdown,
        "currency": "人民币/元",
    }
    logger.info(f"[calc_ticket_price]{cname} {season_name} {platform}: 总价{total_with_fee}元")
    return json.dumps(result, ensure_ascii=False, indent=2)
@tool
def plan_route(scenic_spot: str, traveler_types: str = '["普通成人"]', duration_hours: float = 4.0) -> str:
    """根据景区名称、出行人员类型和预计游览时间，生成分段式游览路线规划，包含景点节点、时间预估、无障碍设施和休息点信息，返回结构化文本。

    入参 scenic_spot 为景区名称，traveler_types 为JSON数组字符串，duration_hours 为数字（小时）。
    """
    try:
        ttypes = json.loads(traveler_types) if isinstance(traveler_types, str) else traveler_types
    except json.JSONDecodeError:
        ttypes = [traveler_types]

    # 模糊匹配景区
    route = None
    for name, r in ROUTE_TEMPLATES.items():
        if scenic_spot.strip() in name or name in scenic_spot.strip():
            route = r
            break

    if not route:
        return json.dumps({
            "message": f"暂时没有{scenic_spot}的预设路线模板，建议参考以下通用建议：游览时间约{duration_hours}小时",
            "available_routes": list(ROUTE_TEMPLATES.keys()),
        }, ensure_ascii=False, indent=2)

    # 检查特殊人群需求
    has_elderly = any("老人" in str(t) for t in ttypes)
    has_wheelchair = any("残疾" in str(t) for t in ttypes) or any("轮椅" in str(t) for t in ttypes)

    segments_output = []
    total_minutes = 0
    for seg in route["segments"]:
        node_minutes = seg["duration_min"]
        if has_elderly:
            node_minutes = int(node_minutes * 1.6)  # 老人慢速调整
        total_minutes += node_minutes
        accessible_flag = "✓" if seg["accessible"] else "✗"
        rest_flag = "有休息点" if seg["rest"] else "无休息点"
        segments_output.append(
            f"  [{seg['node']}] {seg['desc']} | 约{node_minutes}分钟 | 无障碍:{accessible_flag} | {rest_flag}"
        )

    plan_text = f"""
【{route['name']}】路线规划
━━━━━━━━━━━━━━━━━━━━━━
预计总时长：约{round(total_minutes / 60, 1)}小时（{total_minutes}分钟）
路线节点：
{chr(10).join(segments_output)}
━━━━━━━━━━━━━━━━━━━━━━
"""
    if has_elderly:
        plan_text += f"\n老人友好提示：{route.get('elderly_note', '建议放慢节奏，适当增加休息时间')}"
    if has_wheelchair:
        plan_text += f"\n无障碍提示：{route.get('wheelchair_note', '请提前确认无障碍通道开放情况')}"

    logger.info(f"[plan_route]{scenic_spot} 路线已生成，{len(route['segments'])}个节点")
    return plan_text


@tool
def guide_order_exec(action: str, context: str = "{}") -> str:
    """执行导览讲解、互动答疑或生成预订单。三种模式：
    - narration：返回景点生动讲解词
    - qa：回答游客互动提问
    - preorder：生成结构化JSON预订单（包含完整票务明细、核验方式、入园说明）

    入参 action 为"narration"/"qa"/"preorder"三者之一，context为相关JSON上下文字符串。
    """
    action_lower = action.strip().lower()

    if action_lower == "narration":
        try:
            ctx = json.loads(context) if isinstance(context, str) else context
        except json.JSONDecodeError:
            ctx = {"scenic_spot": context, "node": ""}

        spot = ctx.get("scenic_spot", "")
        node = ctx.get("node", "")

        # 自动关联点位→所属景区（如"北一楼"→"八达岭长城"）
        parent_spot = POINT_TO_SPOT.get(spot, "")
        if parent_spot and not node:
            node = spot
            spot = parent_spot

        # 查找讲解素材：先精确匹配景区名，再按节点名全局搜索
        spot_narrations = None
        matched_spot_name = ""
        for name, narrs in NARRATION_MATERIALS.items():
            if spot == name or spot in name or name in spot:
                spot_narrations = narrs
                matched_spot_name = name
                break
        if not spot_narrations:
            for name, narrs in NARRATION_MATERIALS.items():
                if node and node in narrs:
                    spot_narrations = narrs
                    matched_spot_name = name
                    break
                if spot in narrs:
                    spot_narrations = narrs
                    matched_spot_name = name
                    break

        # 命中内置素材 → 直接返回，带STOP前缀
        if spot_narrations:
            prefix = "[这是景点讲解的最终结果，请直接回复用户，不要再调用任何工具]\n\n"
            if node and node in spot_narrations:
                return prefix + f"【{matched_spot_name} - {node}】\n{spot_narrations[node]}"
            if spot in spot_narrations:
                return prefix + f"【{matched_spot_name}】\n{spot_narrations[spot]}"
            parts = [f"【{n}】\n{t}" for n, t in spot_narrations.items()]
            return prefix + f"【{matched_spot_name}】的景点讲解：\n\n" + "\n\n".join(parts)

        # 未命中 → 使用 RAG 检索知识库，组合 点位+景区 作为检索词
        search_terms = [spot]
        if node and node != spot:
            search_terms.append(node)
        if parent_spot and parent_spot not in search_terms:
            search_terms.append(parent_spot)
        rag_query = " ".join(search_terms) + " 景点介绍 历史 看点"
        logger.info(f"[guide_order_exec:narration] RAG检索：{rag_query}")
        rag_result = rag.rag_summarize(rag_query)

        return (
            f"[这是通过知识库检索到的讲解内容，请以此为基础回复用户，不要再调用工具]\n\n"
            f"关于 {' '.join(search_terms)} 的讲解：\n\n{rag_result}"
        )

    elif action_lower == "qa":
        try:
            ctx = json.loads(context) if isinstance(context, str) else context
        except json.JSONDecodeError:
            ctx = {"question": context}

        question = ctx.get("question", context)
        logger.info(f"[guide_order_exec:qa]收到提问：{question}")
        return f"感谢您的提问：「{question}」。建议结合景区实际情况，您也可以调用 search_policy 工具检索相关知识库获取更详细的信息。如果您在景区现场，欢迎随时向我咨询景点历史、路线指引、餐饮推荐等问题！"

    elif action_lower == "preorder":
        try:
            ctx = json.loads(context) if isinstance(context, str) else context
        except json.JSONDecodeError:
            ctx = {}

        scenic = ctx.get("scenic_spot", "未知景区")
        platform = ctx.get("platform", "meituan")
        tickets_info = ctx.get("tickets", [])
        total_price = ctx.get("total_price", 0)
        persons = ctx.get("persons", [])

        # 生成预订单
        import datetime
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        prefix_map = {"meituan": "MT", "ctrip": "CTRIP", "spot_self": "SPOT"}
        prefix = prefix_map.get(platform, "PRE")
        order_id = f"PRE{prefix}{now}{random.randint(1000, 9999)}"

        ticket_items = []
        for t in tickets_info if tickets_info else [{"type": "成人票", "quantity": 1, "unit_price": 0, "discount": 0}]:
            ticket_items.append({
                "type": t.get("type", "成人票"),
                "quantity": t.get("quantity", 1),
                "unit_price": t.get("unit_price", 0),
                "discount": t.get("discount", 0),
                "discount_reason": t.get("discount_reason", ""),
                "subtotal": round(t.get("unit_price", 0) * t.get("quantity", 1) - t.get("discount", 0), 2),
            })

        required_docs = set()
        for p in persons:
            ptype = p.get("person_type", "")
            for key, rule in PERSON_TYPE_DISCOUNTS.items():
                if ptype in key or key in ptype:
                    required_docs.update(rule.get("required_docs", []))
                    break

        order = {
            "order_id": order_id,
            "platform": platform,
            "scenic_spot": scenic,
            "tickets": ticket_items,
            "total_price": round(total_price, 2),
            "platform_service_fee": _get_platform_fees().get(platform, {}).get("service_fee", 0),
            "verification_methods": ["身份证", "二维码"] + (["短信验证码"] if platform in ("meituan", "ctrip") else []),
            "required_documents": list(required_docs) if required_docs else ["身份证"],
            "entry_instructions": f"请携带{'、'.join(required_docs) if required_docs else '身份证'}在{scenic}入口闸机处，使用任意一种凭证（身份证/二维码/短信验证码）即可入园。如需帮助请联系景区服务台。",
            "visit_date": ctx.get("visit_date", "请确认游览日期"),
            "order_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info(f"[guide_order_exec:preorder]预订单已生成：{order_id}")
        return json.dumps(order, ensure_ascii=False, indent=2)

    else:
        return f"不支持的操作类型：{action}。支持的操作为：narration（讲解）、qa（答疑）、preorder（预订单生成）"


@tool
def ticket_query(scenic_spot: str, visit_date: str, traveler_count: int = 1) -> str:
    """查询指定景区在指定日期的多平台余票信息，返回各平台票价和剩余数量。

    入参 scenic_spot 为景区名称，visit_date 为游玩日期(YYYY-MM-DD格式)，traveler_count 为出行人数。
    仅在用户明确询问"查票""余票""有没有票""还有票吗"时调用。
    """
    if not scenic_spot:
        return "请提供要查询的景区名称。"
    if not visit_date:
        return "请提供计划游玩的日期（如2026-06-15）。"

    # 日期格式校验
    try:
        datetime.datetime.strptime(visit_date, "%Y-%m-%d")
    except ValueError:
        return f"日期格式不正确：{visit_date}，请使用 YYYY-MM-DD 格式（如 2026-06-15）。"

    try:
        client = get_ticket_client()
        result = client.query_tickets(scenic_spot, visit_date)
    except TicketClientError as e:
        logger.error(f"[ticket_query]查询失败：{e}")
        return f"很抱歉，{scenic_spot}的余票查询暂时不可用（{str(e)}），请稍后重试或通过景区官方渠道查询。"

    # 组装友好的自然语言回复
    platforms = result.get("platforms", [])
    if not platforms:
        return f"{scenic_spot} 在 {visit_date} 暂无余票信息，建议通过景区官方渠道确认。"

    lines = [f"{result['scenic_spot']}  {visit_date}  多平台余票："]
    total_remain = 0
    for p in platforms:
        status_emoji = "✓" if p["status"] == "有票" else "✗"
        adult_price = p["adult_ticket"]["price"]
        child_price = p["child_ticket"]["price"]
        adult_left = p["adult_ticket"]["remaining"]
        child_left = p["child_ticket"]["remaining"]
        total_remain += p["total_remaining"]
        lines.append(
            f"  {status_emoji} {p['platform_name']}：成人票{adult_price}元(余{adult_left}张)，"
            f"儿童票{child_price}元(余{child_left}张)"
        )
    lines.append(f"  合计剩余约{total_remain}张")
    if traveler_count > total_remain:
        lines.append(f"  您需要{traveler_count}张票，当前余票不足，建议尝试其他平台或日期")

    logger.info(f"[ticket_query]{scenic_spot} {visit_date} 查询完成，{len(platforms)}平台")
    return "\n".join(lines)


@tool
def ticket_book(scenic_spot: str, visit_date: str, traveler_count: int = 1, phone: str = "", price_breakdown: str = "") -> str:
    """协助用户完成门票预订/预约。提交预订请求，返回订单确认信息。

    入参 scenic_spot 为景区名称，visit_date 为游玩日期，traveler_count 为出行人数，phone 为11位手机号，
    price_breakdown 为 calc_ticket_price 返回的价格明细 JSON 字符串（可选，传入时按真实优惠价格下单）。
    仅在用户明确表示"订票""预订""买票""帮我订"时调用。
    """
    if not scenic_spot:
        return "请提供要预订的景区名称。"
    if not visit_date:
        return "请提供计划游玩的日期（如2026-06-15）。"
    if not phone:
        return "预订需要提供联系手机号，请告知您的11位手机号码（仅用于接收订单确认短信）。"
    if not isinstance(traveler_count, int) or traveler_count <= 0:
        return "出行人数格式有误，请提供正整数（如1、2、3）。"

    try:
        datetime.datetime.strptime(visit_date, "%Y-%m-%d")
    except ValueError:
        return f"日期格式不正确：{visit_date}，请使用 YYYY-MM-DD 格式（如 2026-06-15）。"

    try:
        client = get_ticket_client()
        result = client.book_ticket(scenic_spot, visit_date, traveler_count, phone, price_breakdown)
    except TicketClientError as e:
        logger.error(f"[ticket_book]预订失败：{e}")
        return f"很抱歉，{scenic_spot}的门票预订暂时不可用（{str(e)}），请稍后重试或通过景区官方渠道预订。"

    logger.info(f"[ticket_book]预订成功：{result['order_id']}")
    masked_phone = f"{phone[:3]}****{phone[-4:]}"
    return (
        f"预订成功！\n"
        f"  订单编号：{result['order_id']}\n"
        f"  景区：{result['scenic_spot']}\n"
        f"  日期：{result['visit_date']}\n"
        f"  人数：{result['traveler_count']}人\n"
        f"  联系手机：{result['phone']}\n"
        f"  预估总价：{result['total_price']}元\n"
        f"  {result['note']}"
    )
