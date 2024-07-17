#!/usr/bin/env python
import subprocess
import time
from threading import Thread, Event
import re
import datetime
import shlex
import rospy

def monitor_power():
    interval = rospy.get_param("~interval", 5)
    p = subprocess.Popen(shlex.split(f"sudo tegrastats --interval {int(interval*1000)}"), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while not rospy.is_shutdown() and p.poll() is None:
        line = p.stdout.readline().decode('utf8')
        line = line.strip()
        try:
            ts = datetime.datetime.strptime(line[:19], "%m-%d-%Y %H:%M:%S").timestamp()
        except:
            rospy.logerr(line)
            continue
        match = re.search(r"VDD_IN(.*)", line, re.DOTALL)
        if match:
            power = line[match.start(): match.end()]
        else:
            power = "None"
        rospy.loginfo(f"timestamp {ts:.2f} {power}")
        time.sleep(interval)
    p2 = subprocess.Popen(shlex.split("sudo tegrastats --stop"), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2.wait()
    p.wait()

rospy.init_node("power_monitor")
rospy.loginfo("Start power_monitor")
monitor_power()
rospy.loginfo("Stop power_monitor")
