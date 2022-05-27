#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <cstdlib>
#include <algorithm>
using namespace std;

struct node
{
	node *leftChild;
	node *rightChild;
	int frequency;	//for number of times of the occurence
	int content;		//label for the ndoe as from the histogram
	string code;		//actual code for the node as from huffman algorithm
	bool operator<(const node &a) const { return (frequency > a.frequency); }
};

class Huffman
{
	priority_queue<node> nodeArray;	//data strored as minheap
	vector<float> codeTable;		//to store the frequency fo each node
	vector<string> symbolTable;		//to store the code(string) for the nodes in min order in priority queue
	vector<int> image;			//vector to store the dct image in vector form

public:
	// Huffman(){};

	//member function to return the rootnode for the huffman tree
	node getHuffmanTree()
	{
		while (!nodeArray.empty())
		{
			node *rootNode = new node;
			node *leftNode = new node;
			node *rightNode = new node;

			*leftNode = nodeArray.top();
			nodeArray.pop();
			*rightNode = nodeArray.top();
			nodeArray.pop();

			rootNode->leftChild = leftNode;
			rootNode->rightChild = rightNode;
			rootNode->frequency = leftNode->frequency + rightNode->frequency;

			nodeArray.push(*rootNode);
			if (nodeArray.size() == 1)
			{ // only the root node exsits
				break;
			}
		}

		return nodeArray.top();
	}

	//Breadth first searching for symbol for the content in the tree
	void BFS(node *temproot, string s)
	{
		node *root1 = new node;
		root1 = temproot;

		root1->code = s;

		if (root1 == NULL)
		{
		}
		else if (root1->leftChild == NULL && root1->rightChild == NULL)
		{
			short i = root1->content;
			codeTable.at(i) = (float)root1->frequency;
			symbolTable.at(i) = root1->code;
		}
		else
		{
			root1->leftChild->code = s.append("0");
			s.erase(s.end() - 1);
			root1->rightChild->code = s.append("1");
			s.erase(s.end() - 1);

			BFS(root1->leftChild, s.append("0"));
			s.erase(s.end() - 1);
			BFS(root1->rightChild, s.append("1"));
			s.erase(s.end() - 1);
		}
	}

	//computes and returns the table of symbols
	vector<float> getHuffmanCode()
	{
		node root = getHuffmanTree(); // construct the huffman tree
		//UCHAR_MAX is from limits header in C to get the max value of an unsigned char object it returns the max size that an unsigned char can store which is 255
		codeTable.resize(UCHAR_MAX + 1);	 // Code table with 256 bins
		symbolTable.resize(UCHAR_MAX + 1); // Code table with 256 bins

		BFS(&root, ""); // Search tree-basead code with Breadth first searching

		return codeTable; // return table of symbols
	}

	// sets up the frequency table to create the huffman tree
	void setFrequenceTable(vector<float> f)
	{
		for (unsigned i = 0; i < f.size(); i++)
		{
			setFrequenceTable(i, f[i]);
		}
	}

	void setFrequenceTable(int ind, float frequency)
	{
		if (frequency <= 0)
			return;

		node temp;
		temp.frequency = (int)frequency;
		temp.content = ind;
		temp.leftChild = NULL;
		temp.rightChild = NULL;
		nodeArray.push(temp);
	}


	// to encode into 0s and 1s based on the optimal prefix coding of huffman coding algorithm
	string encode(vector<int> e)
	{
		string codifiedImage = "";

		for (unsigned i = 0; i < e.size(); i++)
		{
			if (symbolTable.at(e.at(i)) == "")
			{
				cout << "\nError: Code don't exist in CodeTable." << endl;
			}
			codifiedImage += symbolTable.at(e.at(i));
		}
		return codifiedImage;
	}

	// to decode the string
	vector<int> decode(string d)
	{
		node root = getHuffmanTree();

		searchContent(&root, d);

		return image;
	}


	// used to traverse to decode a particular bit pattern in string im passed as argument
	void searchContent(node *root, string im)
	{
		node *n = new node;
		n = root;

		size_t imSize = im.length();
		image.clear();

		for (size_t i = 0; i <= imSize; i++)
		{
			if (n->leftChild == NULL && n->rightChild == NULL)
			{ // leaf
				image.push_back(n->content);
				n = root;
			}

			n = (im[i] == '1') ? n->rightChild : n->leftChild;
		}
		im.clear();
	}

	// virtual ~Huffman(){};
};
