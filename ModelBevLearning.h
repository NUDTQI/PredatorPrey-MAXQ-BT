#pragma once
#include "Prey.h"
#include "AI/rl/EnvModel.h"
#include "AI/rl/Action.h"
#include "AI/rl/CPrimitiveAction.h"
#include "AI/rl/State.h"
#include "StateBevLearning.h"
#include "AI/RL/Evaluator.h"

//该文件需要用户为所有需要学习的行为树节点定义其奖励函数模型，对应所设计的状态类
//由于可以统一的将可选动作定义为子节点的序号，因此动作模型无需实例化


//动作1表示retreat节点；动作2表示attack；动作3表示idle
class ModelSelRootBevLearning
	: public EnvModelLearning
{
public:

	ModelSelRootBevLearning(Prey* pA, Evaluator* pEva)
		: EnvModelLearning(pA,pEva){}

	~ModelSelRootBevLearning(void){}

	StateRootBevSel temp;
	
public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateRootBevSel* ps = new StateRootBevSel();
		ps->CopyState(temp);
		return ps;
	}


	//本函数主要计算第一部分自身定义的状态动作空间的伪奖励值
	virtual float CalculateRewardsEachStep()
	{
		StateRootBevSel* pS = dynamic_cast<StateRootBevSel*>(pState);
		StateRootBevSel* preS = dynamic_cast<StateRootBevSel*>(preState);

		////////////////以下为Root节点空间内定义的单步伪奖励值函数//////////////////////////////////////////////////////////
		float pseudoReward = 0;
		if (NonHl==pS->GetHealthlevel())
		{
			pseudoReward = pseudoReward - 10;
		}
		if (Inside==pS->GetDisToHaven())
		{
			pseudoReward = pseudoReward + 15;
		}

		//没有敌人，没有发现haven选择撤退惩罚
		if (None==pS->GetDisToHaven() && None==pS->GetDisToEnemy() && preAction->getAction()==0)
		{
			pseudoReward = pseudoReward - 0.5;
		}

		//没有敌人，选择进攻惩罚
		if (None==pS->GetDisToEnemy() && preAction->getAction()==2)
		{
			pseudoReward = pseudoReward - 0.5;
		}

		//离敌人不远，选择漫游惩罚
		if (Middle>pS->GetDisToEnemy() && preAction->getAction()==1)
		{
			pseudoReward = pseudoReward - 0.5;
		}

		//血量增加，奖励
		if (pS->GetHealthlevel() > preS->GetHealthlevel())
		{
			pseudoReward = pseudoReward + 2;
		}

		if (LowHl == pS->GetHealthlevel())
		{
			if (Near==pS->GetDisToEnemy() && preAction->getAction()==0)
			{
				pseudoReward = pseudoReward + 0.5;
			}
		}

		//队友多血量高鼓励进攻
		if(HighHl==pS->GetHealthlevel() && LowAlly<pS->GetNumAllyNeighbor() && None>pS->GetDisToEnemy() && preAction->getAction()==2)
		{
			pseudoReward = pseudoReward + 0.5;
		}		
		return pseudoReward;
	}
};

class ModelSelRetreatBevLearning
	: public EnvModelLearning
{
public:

	ModelSelRetreatBevLearning(Prey* pA,Evaluator* pEva)
		: EnvModelLearning(pA,pEva){}

	~ModelSelRetreatBevLearning(void){}

	StateRetreatBevSel temp;

public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateRetreatBevSel* ps = new StateRetreatBevSel();
		ps->CopyState(temp);
		return ps;
	}

	virtual float CalculateRewardsEachStep()
	{
		StateRetreatBevSel* pS = dynamic_cast<StateRetreatBevSel*>(pState);

		////////////////以下为retreat节点空间内定义的单步伪奖励值函数//////////////////////////////////////////////////////////
		float pseudoReward = 0;
		//对于撤退行为，奖励考虑自身的生命情况和到达安全区域的情况
		if (NonHl == pS->GetHealth())
		{
			pseudoReward = pseudoReward - 10;
		}

		if (Inside == pS->GetDisToHaven())
		{
			pseudoReward = pseudoReward + 15;
		}

		//没有Haven，选择到达Haven惩罚
		if (None==pS->GetDisToHaven() && preAction->getAction()==1)
		{
			pseudoReward = pseudoReward - 0.5;
		}
		return pseudoReward;
	}
};

class ModelSelAttackBevLearning
	: public EnvModelLearning
{
public:

	ModelSelAttackBevLearning(Prey* pA,Evaluator* pEva)
		: EnvModelLearning(pA,pEva){}

	~ModelSelAttackBevLearning(void){}

	StateAttackBevSel temp;

public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateAttackBevSel* ps = new StateAttackBevSel();
		ps->CopyState(temp);
		return ps;
	}

	virtual float CalculateRewardsEachStep()
	{
		float pseudoReward = 0; 
		StateAttackBevSel* pS = dynamic_cast<StateAttackBevSel*>(pState);

		////////////////以下为attack节点空间内定义的单步伪奖励值函数//////////////////////////////////////////////////////////
		//float pseudoReward = 0;
		//对于进攻行为，奖励考虑自身的生命情况和到达安全区域的情况

		if (NonHl == pS->GetHealthlevel())
		{
			pseudoReward = pseudoReward - 10;
		}

		if (m_pOwner->GetNearestPredator() && m_pOwner->GetNearestPredator()->isDead())
		{
			pseudoReward = pseudoReward + 2;
		}

		return pseudoReward;
	}
};


//general learning 
class ModelSelGeneralBevLearning
	: public EnvModelLearning
{
public:

	ModelSelGeneralBevLearning(Prey* pA,Evaluator* pEva)
		: EnvModelLearning(pA,pEva){startime = 0;}

	~ModelSelGeneralBevLearning(void){}

	StateGeneralBevSel temp;

public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateGeneralBevSel* ps = new StateGeneralBevSel();
		ps->CopyState(temp);
		return ps;
	}
};