# -*- coding: utf-8 -*-
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dictnum = {}
        for i in range(len(nums)):
            if target-nums[i] in dictnum:
                return [dictnum[target-nums[i]],i]
            dictnum[nums[i]] = i
