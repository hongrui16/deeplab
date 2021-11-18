import pytz
import datetime

# 查询中国所拥有的时区
cn = pytz.country_timezones('cn')
# 查询美国所拥有的时区
us = pytz.country_timezones('us')
# 查询韩国所拥有的时区
kr = pytz.country_timezones('kr')
print(cn)
# ['Asia/Shanghai', 'Asia/Urumqi']
print(us)
# ['America/New_York', 'America/Detroit', 'America/Kentucky/Louisville', 'America/Kentucky/Monticello', 'America/Indiana/Indianapolis', 'America/Indiana/Vincennes', 'America/Indiana/Winamac', 'America/Indiana/Marengo', 'America/Indiana/Petersburg', 'America/Indiana/Vevay', 'America/Chicago', 'America/Indiana/Tell_City', 'America/Indiana/Knox', 'America/Menominee', 'America/North_Dakota/Center', 'America/North_Dakota/New_Salem', 'America/North_Dakota/Beulah', 'America/Denver', 'America/Boise', 'America/Phoenix', 'America/Los_Angeles', 'America/Anchorage', 'America/Juneau', 'America/Sitka', 'America/Metlakatla', 'America/Yakutat', 'America/Nome', 'America/Adak', 'Pacific/Honolulu']
print(kr)
['Asia/Seoul']
# 选择时区，生成一个时区对象,首尔时区
tz = pytz.timezone('Asia/Seoul')

#需要传递一个时区，如果不传，就默认是当前用户所在时区
kr_time_str = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
local_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(kr_time_str)
print(local_time_str)
