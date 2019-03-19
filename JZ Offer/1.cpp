class Solution {
public:
	bool Find(int target, vector<vector<int> > array) {
		int rows = (int)array.size();
		int cols = (int)array[0].size();
		int j = cols - 1;
		int i = 0;
		while (j >= 0 && i <= rows - 1)
		{
			if (array[i][j] > target)
			{
				j--;
			}
			else if (array[i][j] < target)
			{
				i++;
			}
			else
			{
				return 1;
			}
		}
		return 0;
	}
};