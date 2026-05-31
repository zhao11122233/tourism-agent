import os
from utils.logger_handler import logger
from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import urllib.request
import urllib.error
import urllib.parse
import json

rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


@tool
def rag_summarize(query: str) -> str:
    """从向量存储中检索参考资料"""
    return rag.rag_summarize(query)


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气，以消息字符串的形式返回"""
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        humidity = current["humidity"]
        wind_dir = current["winddir16Point"]
        wind_speed = current["windspeedKmph"]
        weather_desc = current["weatherDesc"][0]["value"]
        uv_index = current["uvIndex"]

        return (
            f"城市{city}当前天气：{weather_desc}，气温{temp_c}摄氏度，"
            f"空气湿度{humidity}%，{wind_dir}风{wind_speed}公里/小时，紫外线指数{uv_index}"
        )
    except Exception as e:
        logger.warning(f"[get_weather]获取{city}天气失败：{str(e)}")
        return f"城市{city}天气信息暂时无法获取，请稍后重试"


@tool
def get_user_location() -> str:
    """随机获取用户所在城市的名称，以纯字符串形式返回"""
    return random.choice(["深圳", "北京", "杭州", "上海", "广州", "成都", "武汉", "南京", "重庆", "西安"])


@tool
def get_user_id() -> str:
    """获取用户的ID，以纯字符串形式返回"""
    return random.choice(user_ids)


@tool
def get_current_month() -> str:
    """获取当前月份，以纯字符串形式返回"""
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"基本信息": xxx, "健康指标": xxx, ...}
            "month" : {"基本信息": xxx, "健康指标": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                basic_info: str = arr[1].replace('"', "")
                health_metrics: str = arr[2].replace('"', "")
                exercise: str = arr[3].replace('"', "")
                nutrition: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "基本信息": basic_info,
                    "健康指标": health_metrics,
                    "运动健身": exercise,
                    "饮食营养": nutrition,
                }


@tool
def fetch_external_data(query: str) -> str:
    """从外部系统中获取指定用户在某个月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串。

    query 参数中需包含 user_id 和 month，支持以下格式：
    - "user_id=1001, month=2025-06"
    - "1001, 2025-06"
    - 直接拼在一起如 "user_id=1001 month=2025-06"
    """
    import re

    generate_external_data()

    user_id = ""
    month = ""

    user_match = re.search(r'(?:user_id\s*[=:]\s*)?(\d{4})', query)
    month_match = re.search(r'(?:month\s*[=:]\s*)?(\d{4}-\d{2})', query)

    if user_match:
        user_id = user_match.group(1)
    if month_match:
        month = month_match.group(1)

    if not user_id or not month:
        logger.warning(f"[fetch_external_data]无法从query中解析出user_id和month：{query}")
        return "未能解析用户ID和月份，请提供正确的格式如 user_id=1001, month=2025-06"

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
#if __name__ =='__main__':
#    print=(fetch_external_data("1001","2025-01"))

@tool
def fill_context_for_report():
    """无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"""
    return "fill_context_for_report已调用"
