"""
独立票务客户端层
负责第三方平台 API 对接、数据解析、异常兜底。
Agent 层仅通过本模块访问外部票务数据，不直接发起网络请求。
"""

import json
import os
import random
import time
from datetime import datetime, timedelta
from utils.logger_handler import logger
from utils.config_handler import agent_conf


class TicketClientError(Exception):
    """票务客户端异常"""


# 从统一价格数据文件加载（与 agent_tools.py 共享数据源）
_pricing_data = None

def _load_pricing():
    global _pricing_data
    if _pricing_data is None:
        _pricing_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pricing', 'scenic_spots.json')
        with open(_pricing_path, 'r', encoding='utf-8') as _f:
            _pricing_data = json.load(_f)
    return _pricing_data

def _in_date_range(month_day: str, start: str, end: str) -> bool:
    if start <= end:
        return start <= month_day <= end
    else:
        return month_day >= start or month_day <= end

def _resolve_spot_client(name: str):
    """根据景区名或别名查找景区价格数据"""
    data = _load_pricing()
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


class TicketClient:
    """统一票务查询与预订客户端"""

    def __init__(self):
        client_conf = agent_conf.get("ticket_client", {})
        self.base_url = client_conf.get("base_url", "")
        self.timeout = client_conf.get("timeout", 10)
        self.retry = client_conf.get("retry", 2)

    def _mock_query(self, scenic_spot: str, visit_date: str) -> dict:
        """模拟多平台余票查询，返回美团/携程/景区自营三平台数据（价格从 data/pricing/scenic_spots.json 加载）"""
        cname, spot_data = _resolve_spot_client(scenic_spot)
        spot_matched = cname if cname else scenic_spot

        # 判断淡旺季
        if visit_date and len(visit_date) >= 10:
            month_day = visit_date[5:10]
            peak = spot_data["peak_season"] if spot_data else {"start": "01-01", "end": "12-31"}
            if _in_date_range(month_day, peak["start"], peak["end"]):
                season = "peak"
            else:
                season = "off_peak"
        else:
            season = "peak"

        if spot_data:
            prices = spot_data[f"prices_{season}"]
            base = prices["adult"]
            base_child = prices["child"]
        else:
            base = 80
            base_child = max(base // 2, 0)

        platforms = []
        for pf_name, pf_id in [("美团", "meituan"), ("携程", "ctrip"), ("景区官方", "spot_self")]:
            adult_left = random.randint(0, 30)
            child_left = random.randint(0, 15)
            platforms.append({
                "platform": pf_id,
                "platform_name": pf_name,
                "adult_ticket": {"price": base, "remaining": adult_left},
                "child_ticket": {"price": base_child, "remaining": child_left},
                "total_remaining": adult_left + child_left,
                "status": "有票" if (adult_left + child_left) > 0 else "售罄",
            })

        return {
            "scenic_spot": spot_matched,
            "visit_date": visit_date,
            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "platforms": platforms,
        }

    def _mock_book(self, scenic_spot: str, visit_date: str, traveler_count: int, phone: str) -> dict:
        """模拟门票预订，返回预订确认信息"""
        order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
        return {
            "success": True,
            "order_id": order_id,
            "scenic_spot": scenic_spot,
            "visit_date": visit_date,
            "traveler_count": traveler_count,
            "phone": f"{phone[:3]}****{phone[-4:]}",
            "total_price": 60 * traveler_count,  # 模拟成人票
            "status": "已预订",
            "book_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "请于游览当日凭订单号和身份证入园，如需取消请在游览前3天操作",
        }

    def query_tickets(self, scenic_spot: str, visit_date: str) -> dict:
        """查询多平台余票信息

        Args:
            scenic_spot: 景区名称
            visit_date: 游玩日期 (YYYY-MM-DD)

        Returns:
            dict: 包含各平台余票和价格的查询结果

        Raises:
            TicketClientError: 查询失败时抛出
        """
        if not scenic_spot or not visit_date:
            raise TicketClientError("景区名称和游玩日期不能为空")

        attempt = 0
        last_error = None
        while attempt <= self.retry:
            try:
                logger.info(f"[TicketClient] 查询{scenic_spot} {visit_date} 余票 (第{attempt+1}次)")
                # TODO: 接入真实 API 时替换此处
                if self.base_url:
                    # 预留：真实 HTTP 请求逻辑
                    # import urllib.request
                    # url = f"{self.base_url}/tickets/query?..."
                    # ...
                    pass
                result = self._mock_query(scenic_spot, visit_date)
                logger.info(f"[TicketClient] 查询成功：{len(result['platforms'])}平台")
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt <= self.retry:
                    time.sleep(1)

        raise TicketClientError(f"余票查询失败（已重试{self.retry}次）：{last_error}")

    def book_ticket(self, scenic_spot: str, visit_date: str, traveler_count: int, phone: str) -> dict:
        """提交门票预订

        Args:
            scenic_spot: 景区名称
            visit_date: 游玩日期 (YYYY-MM-DD)
            traveler_count: 出行人数
            phone: 联系手机号 (11位)

        Returns:
            dict: 包含订单号和预订状态的确认信息

        Raises:
            TicketClientError: 预订失败时抛出
        """
        if not scenic_spot or not visit_date:
            raise TicketClientError("景区名称和游玩日期不能为空")
        if not isinstance(traveler_count, int) or traveler_count <= 0:
            raise TicketClientError("出行人数必须为正整数")
        if not phone or len(str(phone)) != 11:
            raise TicketClientError("手机号格式不正确，请输入11位手机号")

        attempt = 0
        last_error = None
        while attempt <= self.retry:
            try:
                logger.info(f"[TicketClient] 预订{scenic_spot} {visit_date} {traveler_count}人 {phone} (第{attempt+1}次)")
                if self.base_url:
                    pass
                result = self._mock_book(scenic_spot, visit_date, traveler_count, phone)
                logger.info(f"[TicketClient] 预订成功：{result['order_id']}")
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt <= self.retry:
                    time.sleep(1)

        raise TicketClientError(f"门票预订失败（已重试{self.retry}次）：{last_error}")


# 模块级单例
_ticket_client: TicketClient | None = None


def get_ticket_client() -> TicketClient:
    """获取 TicketClient 单例"""
    global _ticket_client
    if _ticket_client is None:
        _ticket_client = TicketClient()
    return _ticket_client


if __name__ == '__main__':
    client = TicketClient()
    # 测试查询
    result = client.query_tickets("故宫", "2026-06-15")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 测试预订
    result2 = client.book_ticket("故宫", "2026-06-15", 3, "13812345678")
    print(json.dumps(result2, ensure_ascii=False, indent=2))
