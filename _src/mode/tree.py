import logging
import numpy as np
import heapq
import sys


class Cell:
    def __init__(self, minX: float, maxX: float, points: np.ndarray, depth=0):
        self.xmin = minX
        self.xmax = maxX
        self.points = points  #  The points contained in this cell
        self.depth = depth
        self.num = self.points.shape[0]

    def __lt__(self, other):
        if not isinstance(other, Cell):
            return NotImplemented
        return self.num > other.num  # to extract the maximum value


class EmptyInterval:
    def __init__(self, minX: float, maxX: float, parent_cell: Cell | None = None):
        self.xmin = minX
        self.xmax = maxX
        self.parent_cell = parent_cell

    def __lt__(self, other):
        if not isinstance(other, EmptyInterval):
            return NotImplemented
        return (self.xmax - self.xmin) > (other.xmax - other.xmin) # to extract the maximum value


def split_quad(cell: Cell, empty_cells: list):
    xmid = (cell.xmin + cell.xmax) / 2
    left_child = Cell(
        float(cell.xmin),
        float(xmid),
        cell.points[(cell.points >= cell.xmin) & (cell.points < xmid)],
        cell.depth + 1,
    )
    if left_child.num == 0:
        right_child = Cell(float(xmid), float(cell.xmax), cell.points, cell.depth + 1)
    if left_child.num == cell.num:
        right_child = Cell(float(xmid), float(cell.xmax), np.array([]), cell.depth + 1)
    right_child = Cell(
        float(xmid),
        float(cell.xmax),
        cell.points[(cell.points >= xmid)],
        cell.depth + 1,
    )
    return [left_child, right_child]


def tree_patrition(
    points: np.ndarray, max_cells=100, min_points=2, min_width=0.25, bounds=None
):
    if bounds is None:
        xmin, xmax = points.min(), points.max()
        bounds = (xmin, xmax)
    root = Cell(bounds[0], bounds[1], points)
    queue = [root]
    heapq.heapify(queue)  # append to queue
    leaves = []
    empty_cells = []
    heapq.heapify(empty_cells)  # append to queue
    while queue and len(leaves) + len(queue) < max_cells:
        cell = heapq.heappop(queue)
        if len(cell.points) > min_points:
            children = split_quad(cell, empty_cells)
            num = 0
            for child in children:
                num += child.num
                if child.xmax - child.xmin < 2 * min_width:
                    leaves.append(child)  # append to leaf
                elif child.num == 0:
                    leaves.append(child)  # append to leaf
                    xmid = (child.xmin + child.xmax) / 2
                    heapq.heappush(empty_cells, EmptyInterval(child.xmin, xmid, child))
                    heapq.heappush(empty_cells, EmptyInterval(xmid, child.xmax, child))
                elif child.num <= min_points:
                    # print(f"append leaf {child.xmin}, {child.xmax}, {child.num}")
                    leaves.append(child)  # append to leaf
                    min_point, max_point = np.min(child.points), np.max(child.points)
                    if max_point - min_point > min_width:
                        empty_max = min_point
                        empty_min = max_point
                    else:
                        rate = 0.9
                        empty_max = (
                            np.min(child.points) * rate + (1 - rate) * child.xmin
                        )
                        empty_min = (
                            np.max(child.points) * rate + (1 - rate) * child.xmax
                        )
                    heapq.heappush(
                        empty_cells, EmptyInterval(child.xmin, empty_max, child)
                    )
                    heapq.heappush(
                        empty_cells, EmptyInterval(empty_min, child.xmax, child)
                    )
                else:
                    heapq.heappush(queue, child)
            if num != cell.num:
                logging.info(f"({cell.xmin}, {cell.xmax}) : {num=} {cell.num=}")
                sys.exit()
        else:
            leaves.append(cell)
    leaves.extend(queue) # The remainder shall be treated as leaves.

    while len(leaves) < max_cells: # If max_cells is not reached, sample empty intervals with no data points.
        if len(empty_cells) == 0:
            break
        interval: EmptyInterval = heapq.heappop(empty_cells)
        if interval.parent_cell is not None:
            cell = interval.parent_cell
            # print(f"interval: {interval.xmin}, {interval.xmax}")
            if cell.xmin == interval.xmin and cell.xmax == interval.xmax:
                if interval.xmax - interval.xmin > 2 * min_width: # If there are sections to be divided,
                    xmid = (interval.xmin + interval.xmax) / 2
                    heapq.heappush(
                        empty_cells, EmptyInterval(interval.xmin, xmid, cell)
                    )
                    heapq.heappush(
                        empty_cells, EmptyInterval(xmid, interval.xmax, cell)
                    )
                continue
            elif cell.xmin == interval.xmin:
                cell.xmin = interval.xmax # Trim the empty section starting from the left end
            elif cell.xmax == interval.xmax:
                cell.xmax = interval.xmin # Trim the empty section starting from the right end
        new_cell = Cell(interval.xmin, interval.xmax, np.array([]), depth=0)
        leaves.append(new_cell)
        if interval.xmax - interval.xmin > 2 * min_width:
            xmid = (interval.xmin + interval.xmax) / 2
            heapq.heappush(empty_cells, EmptyInterval(interval.xmin, xmid, new_cell))
            heapq.heappush(empty_cells, EmptyInterval(xmid, interval.xmax, new_cell))
    # clean result
    bounds = []
    num = 0
    nums = []
    for leaf in leaves:
        if leaf.xmax == leaf.xmin:
            continue
        num += leaf.num
        nums.append(leaf.num)
        bounds.append([leaf.xmin, leaf.xmax])
    assert num == points.shape[0]
    bounds, inverse_indices = np.unique(bounds, axis=0,  return_inverse=True)
    nums = np.array(nums)
    nums = np.bincount(inverse_indices, weights=nums)
    return bounds, np.array(nums)
