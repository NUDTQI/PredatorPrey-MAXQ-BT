#include "QLearning.h"
#include "EnvModel.h"
#include "State.h"
#include "Action.h"
#include "../../Agent.h"


QLearning::QLearning(/*Prey* owner, */float gamma_in, float beta_in, float exploration_in)
{
	gamma = gamma_in;
	beta = beta_in;
	exploration = exploration_in;

	//m_pOwner = owner;
	curIter = 0;
	maxIter = 100000;
	

	pioQtable = new QLearningDataIO(&QFunctions,&StateList);
}

QLearning::~QLearning(void)
{
	for (std::map <CAction*, QFunction*>::iterator iter=QFunctions.begin();iter!=QFunctions.end();iter++)
	{
		QFunction* qfs = iter->second;
		qfs->qf.clear();
		qfs = NULL;
	}

	for (std::map <CAction*, QFunction*>::iterator iter=CFunctions.begin();iter!=CFunctions.end();iter++)
	{
		QFunction* qfs = iter->second;
		qfs->qf.clear();
		qfs = NULL;
	}

	for (int i=0;i<ActionList.size();i++)
	{
		CAction* act = ActionList[i];
		delete act;
		act = NULL;
	}
	
	//for (std::vector<CState*>::iterator iter=StateList.begin();iter!=StateList.end();iter++)
	//{
	//	CState* st = *iter;
	//	delete st;
	//	st = NULL;
	//}

	delete pioQtable;
	pioQtable = NULL;

	pEnvModel = NULL;
}

void QLearning::setParameters(float newgamma, float newbeta, float newexploration) 
{
	gamma = newgamma;
	beta = newbeta;
	exploration = newexploration;
}

void QLearning::setActionSet( std::vector<CAction*>& ActionSet)
{
	int ActionNum = ActionSet.size();

	for (std::vector<CAction*>::iterator it=ActionSet.begin();it!=ActionSet.end();it++)
	{
		ActionList.push_back(*it);

		QFunction* qf = new QFunction();
		QFunctions.insert(std::make_pair(*it,qf));
		
		//generate CFunction
		QFunction* cf = new QFunction();
		CFunctions.insert(std::make_pair(*it,cf));
	}
}

void QLearning::setStateSet( std::vector<CState*>& StateSet)
{
	for (std::vector<CState*>::iterator it=StateSet.begin();it!=StateSet.end();it++)
	{
		StateList.push_back(*it);
		bestActionID.insert(std::make_pair(*it,0));

		for(std::map <CAction*, QFunction*>::iterator iter = QFunctions.begin();iter!=QFunctions.end();iter++)
		{
			QFunction* vf = iter->second;
			vf->qf.insert(std::make_pair(*it,0));
		}
	}
}

CAction* QLearning::chooseAction(Agent* owner,CState* state)
{
	double r = RandFloat(); 

	if (r < exploration)
	{
		int act = getRandomRLAction(StateList.size());
		pEnvModel->pAction = ActionList[act];
	}
	else
	{
		CAction* pbestA = getBestQValueOnState(state);
		if (pbestA->getAction() >= 0)
		{
			pEnvModel->pAction = pbestA;
		}
	}
	return pEnvModel->pAction;
}

CAction* QLearning::getBestQValueOnState(CState* state)
{
	double qValue = -1*MaxDouble;

	CAction *pAction = NULL;

	for(std::map <CAction*, QFunction*>::iterator it = QFunctions.begin();it!=QFunctions.end();it++)
	{
		QFunction* vf = it->second;

		if (vf->getQValue(state) > qValue)
		{
			qValue = vf->getQValue(state);
			pAction = it->first;
		}
	}
	
	return pAction;
}

float QLearning::getQValue(CState* state, CAction* action,bool isCompletefunction)
{
	float value = 0.0;

	//adapt to maxq method
	if(false == isCompletefunction)
	{
		//assert (QFunctions[action]);

		value = QFunctions[action]->getQValue(state);
	}
	else
	{
		//assert (CFunctions[action]);

		value = CFunctions[action]->getQValue(state);
	}

	return value;
}

void QLearning::setQValue(CState* state,CAction* action, float v,bool isCompletefunction)
{
	float value = 0.0;

	//adapt to maxq method
	if(false == isCompletefunction)
	{
		//assert (QFunctions[action]);

		(QFunctions[action])->setQValue(state, v);
	}
	else
	{
		//assert (CFunctions[action]);

		(CFunctions[action])->setQValue(state, v);
	}
}

float QLearning::getVValue(CState* state)
{
	//assert (VFunction.getQValue(state));

	return VFunction.getQValue(state);
}

void QLearning::setVValue(CState* state, float v)
{
	VFunction.setQValue(state,v);
}

float QLearning::EvaluateMaxNode(int actionType, QLearning* pQ, CState* pS)
{
	float maxq = 0;

	//primitive action
	if(actionType == 0)
	{
		maxq = pQ->VFunction.getQValue(pS);
	}
	else
	{
		int actionIndex = -1;

		for(int i=0;i<pQ->ChildrenLearners.size();i++)
		{
			QLearning* pChild = pQ->ChildrenLearners[i];

			if(pQ->r_ChildrenLearners.find(pChild) != pQ->r_ChildrenLearners.end())
			{
				actionIndex = pQ->r_ChildrenLearners.find(pChild)->second;
			
				CAction* pA = pQ->ActionList[actionIndex];

				CState* mapState = pChild->findStateinList(pS);

				if (NULL == mapState)
				{
					mapState = pChild->pEnvModel->MemyCopyState(pS);
					pChild->StateList.push_back(mapState);
				}
				float vr = EvaluateMaxNode((int)pA->getActionType(),pChild,mapState);

				float cvalue = pQ->getQValue(pS,pA,true);

				pQ->setQValue(pS,pA,vr+cvalue,false);
			}
		}
		pQ->UpdateVFunction(actionType, pS,0);
		maxq = pQ->getVValue(pS);
	}

	return maxq;
}

void QLearning::updateQValue(int actionType, CState* pState, CAction* pAction, float reward, CState* pNextState, CAction* pNextAction, bool isfinished, QLearning* pChild)
{

	this->EvaluateMaxNode(actionType,this,pNextState);

	UpdateCompleteFunction(actionType,pState,pAction,pNextState,pNextAction,pChild);

}


CState* QLearning::updateEnvModelForRL(Agent* owner)
{
	CState* ps = NULL;
	ps = pEnvModel->UpdateCurState();
	CState* pret = NULL;
	pret = findStateinList(ps);
	if (NULL == pret)
	{
		pret = pEnvModel->MemyCopyState(ps);
		StateList.push_back(pret);
	}
	pEnvModel->pState = pret;

	return pEnvModel->pState;
}

CState* QLearning::findStateinList(CState* state)
{
	if (StateList.empty())
	{
		return NULL;
	}
	else
	{
		for(std::vector <CState*>::iterator it = StateList.begin();it!=StateList.end();it++)
		{
			if ((*it)->IsSameState(state))
			{
				return *it;
			}
		}
		return NULL;
	}
}

int   QLearning::getActionIndex(CAction* action)
{
	int index = -1;
	for(std::vector <CAction*>::iterator it = ActionList.begin();it!=ActionList.end();it++)
	{
		index++;
		if ((*it)->isSameAction(action))
		{
			break;
		}
	}
	return index;
}

int QLearning::selectActionByRL(Agent* owner)
{
	pEnvModel->pAction = chooseAction(owner,pEnvModel->pState);
	
	pEnvModel->bindActionData(pEnvModel->pAction,&pEnvModel->localActionData);

	pEnvModel->pAction->setDuration(0);
	pEnvModel->pAction->setisFinished(false);

	pEnvModel->rewardFeedback = 0;

	// here must make sure the accordance of action index in actionlist and BT  
	int ret = getActionIndex(pEnvModel->pAction);
	//int s = pEnvModel->pState->PrintStateValues()[3];
	return ret;
}

int QLearning::selectActionByBT(Agent* owner,int index)
{
	assert(index<ActionList.size());

	pEnvModel->pAction = ActionList[index];
	
	pEnvModel->bindActionData(pEnvModel->pAction,&pEnvModel->localActionData);

	pEnvModel->pAction->setDuration(0);
	pEnvModel->pAction->setisFinished(false);

	pEnvModel->rewardFeedback = 0;

	return index;
}

void QLearning::AccuRewardsOption(bool actioNode,Agent* owner,int interval)
{
	//执行了一次，动作持续时间增加
	//pEnvModel->bindActionData(pEnvModel->pAction,&pEnvModel->localActionData);
	pEnvModel->bindActionData(pEnvModel->preAction,&pEnvModel->preActionData);

	pEnvModel->preAction->setDuration(interval);

	updateEnvModelForRL(owner);

	//执行动作获得的累计奖励，此处会有差别：
	//基于Option的方法细化每一层次学习器的奖励值，因此执行一步CalculateRewardsEachStep，返回自身奖励；
	//而HSMQ方法的上层奖励由下层累积获得，因此执行一步CalculateRewardsEachStep函数获得下级传回的奖励

	//奖励值可以包括两个层次：
	//一是定义自身的奖励函数空间，获得每一步的伪奖励值
	//二是每一步从下级执行的节点处返回的奖励值
	//如果是原子动作，则下级执行节点返回的奖励值为0，其所获得的伪奖励值即作为单步整棵树返回的reward siginal，记录在tree的output共同使用
	//如果非原子动作，则取得tree的output中底层原子动作单步执行的奖励，进行累积

	float pseudoreward = 0;

	if(true == actioNode)
	{
		pEnvModel->rewardFeedback = pEnvModel->rewardFeedback + pEnvModel->CalculateRewardsEachStep()* pow(gamma,pEnvModel->preAction->getDuration()-1);
	}
}

void QLearning::RecordMemory(Agent* owner)
{
	pEnvModel->preAction = pEnvModel->pAction;
	pEnvModel->preActionData = pEnvModel->localActionData;
	pEnvModel->bindActionData(pEnvModel->preAction,&pEnvModel->preActionData);
	pEnvModel->preState = pEnvModel->pState;
}

void QLearning::KnowledgeLearn(Agent* owner)
{
	// update Q table
	pEnvModel->preAction->setisFinished(true);

	//for debug and control experiment
	int ret = pEnvModel->preState->PrintStateValues()[3];
	int act = pEnvModel->preAction->getAction();
	///

	//updateValue(pEnvModel->preState, pEnvModel->preAction, pEnvModel->rewardFeedback, pEnvModel->pState, pEnvModel->pAction,true);

	//for debug and control experiment
	curIter ++;
}

void QLearning::saveQTableToFile( char* filename)
{
	pioQtable->SaveData(filename);
}

void QLearning::loadQTableFromFile( char* filename)
{
	pioQtable->LoadData(filename);
}

void QLearning::UpdateVFunction(int actiontype, CState* ps, float r)
{
	float val = getVValue(ps);
	
	//primitive action
	if(0 == actiontype)
	{
		setVValue(ps,val+ beta*(r - val));
	}	
	else if(1 == actiontype)//selector composite node, get best Q value
	{
		setVValue(ps,getQValue(ps, getBestQValueOnState(ps),false));
	}
	else if(2 == actiontype)//sequence composite node, get the first child node Q value
	{
		setVValue(ps,getQValue(ps, ActionList[0],false));
	}
	else if(3 == actiontype)//parallel composite node, get sum of all the children node Q value(equal to V value)
	{
		float sum = 0;
		for(int i=0;i<ActionList.size();i++)
		{
			sum = sum + getQValue(ps, ActionList[i],false);
		}
		setVValue(ps,sum);
	}
	
}

void QLearning::UpdateCompleteFunction(int actionType,CState* preState, CAction* preAction, CState* pState, CAction* pNextAction, QLearning* pChild)
{
	std::map<CAction*, QFunction*>::iterator it = CFunctions.find(preAction);
	QFunction* pqf = NULL;
	if(it!=CFunctions.end())
	{
		pqf = it->second;
	}
	assert(pqf);

	//make use of child node's V value to update own complete function
	float nextv = 0;
	if(actionType==0 || actionType==3)//action, parallel or the last sequence or selector node without learning
	{
		pNextAction = NULL;
		nextv = 0;
	}
	else if(actionType==2)//sequence node or selector node without learning, use next node Q to update otherwise V.
	{
		int nextact = preAction->getActionIndex() + 1;
		if(nextact >= this->ActionList.size())
		{
			pNextAction = NULL;
			nextv = 0;
		}
		else
		{
			pNextAction = this->ActionList[nextact];
			nextv = getQValue(pState,pNextAction,false);
		}
	}
	else if(actionType == 1)
	{
		nextv = getVValue(pState);
	}
	
	float v = pqf->getQValue(preState);
	
	int dur = pChild->pEnvModel->durationCount;
	for(int i=0;i<pChild->pEnvModel->seq.size();i++)
	{
		CState* ps = pChild->pEnvModel->seq[i];
		if(i-1 >= 0)
		{
			dur = dur - pChild->pEnvModel->seqDur[i-1];
		}
		float newc = (1-beta)*pqf->getQValue(ps) + beta*pow(gamma,dur)*nextv;
		pqf->setQValue(ps,newc);
	}
}

bool QLearning::terminatedTask(CState* pState)
{
	bool finished = false;
	//if(this->LearnerName=="RootLearner")
	//{
	//	StateGeneralBevSel* pS = dynamic_cast<StateGeneralBevSel*>(pState);
	//	if(pS->GetDisToHaven()==DisLevel::Inside || pS->GetHealthlevel()==HealthLevel::NonHl)
	//	{
	//		finished = true;
	//	}
	//}
	if(this->LearnerName=="RetreatLearner")
	{
		StateGeneralBevSel* pS = dynamic_cast<StateGeneralBevSel*>(pState);
		if(pS->GetDisToHaven()==DisLevel::Inside || pS->GetHealthlevel()==HealthLevel::NonHl || 
			(pS->GetDisToEnemy()==DisLevel::None&&pS->GetDisToHaven()==DisLevel::None))
		{
			finished = true;
		}
	}
	else if(this->LearnerName=="IdleLearner")
	{
		StateGeneralBevSel* pS = dynamic_cast<StateGeneralBevSel*>(pState);
		if(pS->GetDisToHaven()==DisLevel::Inside || pS->GetHealthlevel()==HealthLevel::NonHl 
			|| pS->GetDisToHaven()<DisLevel::None || pS->GetDisToEnemy()<DisLevel::None)
		{
			finished = true;
		}
	}
	else if(this->LearnerName=="AttackLearner")
	{
		StateGeneralBevSel* pS = dynamic_cast<StateGeneralBevSel*>(pState);
		if(pS->GetDisToHaven()==DisLevel::Inside || pS->GetHealthlevel()==HealthLevel::NonHl 
			|| pS->GetDisToEnemy()==DisLevel::None|| pS->GetDisToHaven()<DisLevel::None)
		{
			finished = true;
		}
	}
	else if(this->LearnerName=="ExploreQLearner")
	{
		StateGeneralBevSel* pS = dynamic_cast<StateGeneralBevSel*>(pState);
		if(pS->GetDisToHaven()==DisLevel::Inside || pS->GetHealthlevel()==HealthLevel::NonHl 
			|| pS->GetDisToEnemy()<DisLevel::None|| pS->GetDisToHaven()<DisLevel::None)
		{
			finished = true;
		}
	}
	return finished;
}