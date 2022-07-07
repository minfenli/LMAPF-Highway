#!/usr/bin/env python3
from audioop import reverse
from re import S
import yaml
import matplotlib
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.animation as manimation
import argparse
import math

Colors = ['orange', 'blue', 'green']


class Animation:
  def __init__(self, map, schedule, time_limit=None):
    self.map = map
    self.schedule = schedule
    self.combined_schedule = {}
    self.combined_schedule.update(self.schedule["schedule"])
    self.combined_goal = {}
    self.finished_tasks = 0
    if "goal" in self.schedule:
      self.combined_goal.update(self.schedule["goal"])
    self.combined_direction = {}
    if "direction" in self.schedule:
      self.combined_direction.update(self.schedule["direction"])

    aspect = map["map"]["dimensions"][0] / map["map"]["dimensions"][1]

    self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=None, hspace=None)
    # self.ax.set_frame_on(False)

    self.patches = []
    self.arrows = []
    self.artists = []
    self.agents = dict()
    self.agent_names = dict()
    self.agent_goals = dict()
    # create boundary patch
    xmin = -0.5
    ymin = -0.5
    xmax = map["map"]["dimensions"][0] - 0.5
    ymax = map["map"]["dimensions"][1] - 0.5

    # self.ax.relim()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # self.ax.set_xticks([])
    # self.ax.set_yticks([])
    # plt.axis('off')
    # self.ax.axis('tight')
    # self.ax.axis('off')

    self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
    for o in map["map"]["obstacles"]:
      x, y = o[0], o[1]
      self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='red', edgecolor='red'))
    if self.combined_direction:
      for c in map["map"]["corridor"]:
        start, end = c["start"], c["end"]
        self.arrows.append(FancyArrow(start[0] , start[1], math.fabs(start[0] - end[0]), math.fabs(start[1] - end[1]), edgecolor = 'None', width=0.3, head_width = 0.6, head_length = 0.3, alpha = 0.3))
      

    # create agents:
    self.T = 0
    # draw goals first
    # for d, i in zip(map["agents"], range(0, len(map["agents"]))):
    #   self.patches.append(Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.5, 0.5, facecolor='white', edgecolor='none', alpha=1))
    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      name = d["name"]
      self.agents[name] = Circle((d["start"][0], d["start"][1]), 0.3, facecolor=Colors[0], edgecolor='black')
      self.agents[name].original_face_color = Colors[0]
      self.patches.append(self.agents[name])
      self.T = max(self.T, schedule["schedule"][name][-1]["t"])
      self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
      self.agent_names[name].set_horizontalalignment('center')
      self.agent_names[name].set_verticalalignment('center')
      self.artists.append(self.agent_names[name])
      
      self.agent_goals[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''), {'color':'green'})
      self.agent_goals[name].set_horizontalalignment('center')
      self.agent_goals[name].set_verticalalignment('center')
      self.artists.append(self.agent_goals[name])

    self.finished_tasks_text = self.ax.text(xmax - 1, ymax - 1, str(self.finished_tasks), {'backgroundcolor':'grey'})
    self.finished_tasks_text.set_horizontalalignment('right')

    # self.ax.set_axis_off()
    # self.fig.axes[0].set_visible(False)
    # self.fig.axes.get_yaxis().set_visible(False)

    # self.fig.tight_layout()
    
    if time_limit:
        self.T = min(self.T, time_limit)

    self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                               init_func=self.init_func,
                               frames=int(self.T+1) * 10,
                               interval=100,
                               blit=True)

  def save(self, file_name, speed):
    self.anim.save(
      file_name,
      "ffmpeg",
      fps=10 * speed,
      dpi=200),
      # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

  def show(self):
    plt.show()

  def init_func(self):
    for p in self.patches:
      self.ax.add_patch(p)
    for a in self.artists:
      self.ax.add_artist(a)
    for a in self.arrows:
      self.ax.add_patch(a)
    return self.patches + self.artists + self.arrows

  def animate_func(self, i):
    for agent_name, agent in self.combined_schedule.items():
      pos = self.getState(i / 10, agent)
      p = (pos[0], pos[1])
      self.agents[agent_name].center = p
      self.agent_names[agent_name].set_position(p)
    if self.combined_goal:
      for agent_name, agent in self.combined_goal.items():
        pos = self.getGoal(i / 10, agent)
        p = (pos[0], pos[1])
        if self.agent_goals[agent_name].get_position() != p and not i==0:
          self.finished_tasks += 1
          self.finished_tasks_text.set_text(str(self.finished_tasks))
        self.agent_goals[agent_name].set_position(p)
    if self.combined_direction:
      for idx, a in enumerate(self.arrows):
        reverse = self.getDirection(i / 10, self.combined_direction[idx])
        if(not reverse and (a._dx < 0 or a._dy < 0)):
          a.set_data(x=a._x+a._dx, y=a._y+a._dy, dx=-a._dx, dy=-a._dy)
        if(reverse and (a._dx > 0 or a._dy > 0)):
          a.set_data(x=a._x+a._dx, y=a._y+a._dy, dx=-a._dx, dy=-a._dy)

    # reset all colors
    for _,agent in self.agents.items():
      agent.set_facecolor(agent.original_face_color)

    # check drive-drive collisions
    agents_array = [agent for _,agent in self.agents.items()]
    for i in range(0, len(agents_array)):
      for j in range(i+1, len(agents_array)):
        d1 = agents_array[i]
        d2 = agents_array[j]
        pos1 = np.array(d1.center)
        pos2 = np.array(d2.center)
        if np.linalg.norm(pos1 - pos2) < 0.7:
          d1.set_facecolor('red')
          d2.set_facecolor('red')
          print("COLLISION! (agent-agent) ({}, {})".format(i, j))
    

    return self.patches + self.artists + self.arrows + [self.finished_tasks_text]


  def getState(self, t, d):
    idx = 0
    while idx < len(d) and d[idx]["t"] <= t:
      idx += 1
    if idx == 0:
      return np.array([float(d[0]["x"]), float(d[0]["y"])])
    elif idx < len(d):
      posLast = np.array([float(d[idx-1]["x"]), float(d[idx-1]["y"])])
      posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
    else:
      return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
    t = d[idx]["t"] - t
    if t <= 1:
      pos = (posLast - posNext) * t + posNext
    else:
      pos = posLast
    return pos

  def getGoal(self, t, d):
    temp = 0
    for idx, i in enumerate(d):
      if(i["t"] <= t):
        temp = idx
    return np.array([float(d[temp]["x"]), float(d[temp]["y"])])
  
  def getDirection(self, t, d):
    temp = 0
    for idx, i in enumerate(d):
      if(i["t"] <= t):
        temp = idx
    return d[temp]["d"]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("map", help="input file containing map")
  parser.add_argument("schedule", help="schedule for agents")
  parser.add_argument('--video', dest='video', default=None, help="output video file (or leave empty to show on screen)")
  parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
  parser.add_argument("--time_limit", type=int, default=None, help="time limit")
  args = parser.parse_args()


  with open(args.map) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)

  with open(args.schedule) as states_file:
    schedule = yaml.load(states_file, Loader=yaml.FullLoader)

  animation = Animation(map, schedule, args.time_limit)

  if args.video:
    animation.save(args.video, args.speed)
  else:
    animation.show()
