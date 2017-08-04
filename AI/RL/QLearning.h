#pragma once
#include "../../Common/misc/utils.h"
#include "QFunction.h"
#include "QLearningDataIO.h"

class EnvModelLearning;
class Agent;
class CState;
class CAction;
class CActionData;

class QLearning
{
public:
	QLearning(float gamma_in, float beta_in, float exploration_in);
	~QLearning(void);

public:
	float gamma;//折扣因子
	float beta;//学习率
	float exploration;//搜索率

public:
	EnvModelLearning* pEnvModel;
	QLearningDataIO* pioQtable;
	string LearnerName;
	

	std::map <CAction*, QFunction*> QFunctions;
	std::map <CAction*, QFunction*> PsudoQFunctions;
	//expand to maxq learning
	std::map<CAction*, QFunction*> CFunctions;//complete function
	std::map<CAction*, QFunction*> PsudoCFunctions;//pesudo complete function
	QFunction VFunction; //need propogating to parent node

	//for debug output
	std::map<CState*,int> bestActionID;
	int maxIter;
	int curIter;

	std::map<QLearning*,int> r_ChildrenLearners;
	std::map<int,QLearning*> ChildrenLearners;
	QLearning* pParent;
public:
	std::vector<CAction*> ActionList;
	std::vector<CState*> StateList;

	float getVValue(CState* state);
	float getQValue(CState* state, CAction* action,bool isPsudo);
	float getCValue(CState* state, CAction* action,bool isPsudo);
	void  setQValue(CState* state,CAction* action, float v,bool isPsudo);
	void  setCValue(CState* state,CAction* action, float v,bool isPsudo);
	void  setVValue(CState* state, float v);

	int   getActionNum(){return ActionList.size();}
	int   getActionIndex(CAction* action);
	int getRandomRLAction(int ActionNum){return (int)(RandFloat()*ActionList.size());}

	CAction* chooseAction(Agent* owner,CState* state);
	CState* findStateinList(CState* state);
	CAction* getBestQValueOnState(CState* state,bool isPsudo);
	float EvaluateMaxNode(int actionType, QLearning* pQ, CState* pS);
	
public:
	void updateQValue(int actionType,CState* pState, CAction* pAction, float reward, CState* pNextState, CAction* pNextAction, bool isfinished, QLearning* pChild); 
	void UpdateVFunction(int actiontype, CState* ps, float r);
	void UpdateCompleteFunction(int actionType,CState* preState, CAction* preAction,CState* pState, CAction* pNextAction, QLearning* pChild, bool isfinished, float pesudoR);

	void saveQTableToFile(char*);
	void loadQTableFromFile(char*);

	void setParameters(float, float, float);
	//set different action set for each q-learner
	void setActionSet(std::vector<CAction*>&);
	void setStateSet( std::vector<CState*>&);

	//主要接口
	int selectActionByRL(Agent* owner);
	int selectActionByBT(Agent* owner,int index);
	void KnowledgeLearn(Agent* owner);
	void AccuRewardsOption(bool actioNode, Agent* owner,int interval);
	CState* updateEnvModelForRL(Agent* owner);
	void RecordMemory(Agent* owner);
	bool terminatedTask(CState* pState);
};
