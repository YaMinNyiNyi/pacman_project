# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Ghost Agent game states"

from game import Directions, Actions
import util
from util import manhattanDistance

class GhostFeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class GhostIdentityExtractor(GhostFeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

def pacmanDistance(ghost_pos,pacman_pos,walls):
    fringe = [(ghost_pos[0], ghost_pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find pacman at this location then exit
        if (pos_x, pos_y) == (int(pacman_pos[0]), int(pacman_pos[1])):
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None


class GhostAdvancedExtractor(GhostFeatureExtractor):
    
    
    def getFeatures(self, state, action):
      
        # compute the location of ghost after he takes the action
        x, y = state.getGhostPosition(2)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        pacmanPosition = state.getPacmanPosition()
        walls = state.getWalls()
        features = util.Counter()
        features["bias"] = 1.0

        ghostState = state.getGhostState(2)
        isScared = ghostState.scaredTimer > 0

        if(not isScared):
            features["eats-pacman"] = 1.0

        dist = pacmanDistance((next_x, next_y),pacmanPosition, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-pacman"] = float(dist) / (walls.width * walls.height)
        
        capsules = state.getCapsules()
        # calculate the distance between pacman and capsule
        distancesToCapsules = [manhattanDistance( c, pacmanPosition) for c in capsules]
        if(min(distancesToCapsules)<3 or isScared):
               features["eats-pacman"] = 0.0

        features.divideAll(10.0)
        return features





