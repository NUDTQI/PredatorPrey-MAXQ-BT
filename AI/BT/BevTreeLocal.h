#pragma once
#include "BT\BevTreeBasic.h"
#include "2D\Vector2D.h"
#include "../../Prey.h"
#include <string>
#include "../rl/QLearning.h"
#include "../../ModelBevLearning.h"

using namespace  std;
//////////////////////////////////////////////////////////////////////////

class BevInputData 
{
public:
	BevInputData(){}
	~BevInputData(){}

	Prey*    m_pOwner;
};
class BevOutputData
{
public:
	BevOutputData(){}
	~BevOutputData(){}

	std::string m_curStatus;
	//for HSMQ, it represent the accumulated immediate reward;
	//for MAXQ, it represent the V value from the child
	float rewardPropogation;
};

class CON_HasFoodZone : public BevNodePreconditionTRUE
{
public:
	virtual bool ExternalCondition(const BevNodeInputParam& input) const
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		if (prey->GetNearestFood())
		{
			return true;
		}
		return false;
	}
};

class CON_HasHavenZone : public BevNodePreconditionTRUE
{
public:
	virtual bool ExternalCondition(const BevNodeInputParam& input) const
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		if (prey->GetNearestHaven())
		{
			return true;
		}
		return false;
	}
};

class CON_HasEnemy : public BevNodePreconditionTRUE
{
public:
	virtual bool ExternalCondition(const BevNodeInputParam& input) const
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		if (prey->GetNearestPredator())
		{
			return true;
		}
		return false;
	}
};

class CON_InFoodZone : public BevNodePrecondition
{
public:
	virtual bool ExternalCondition(const BevNodeInputParam& input) const
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		if (prey->InFoodZone())
		{
			return true;
		}
		return false;
	}
};

class CON_NeedAssist : public BevNodePrecondition
{
public:
	virtual bool ExternalCondition(const BevNodeInputParam& input) const
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		if (prey->Health()<20 && prey->getNumAllyNeighbour()>0)
		{
			return true;
		}
		return false;
	}
};

class CON_RandomGreedyActive : public BevNodePrecondition
{
public:
	CON_RandomGreedyActive(double exploration, double leftProb, double rightProb,bool isMaxWeightNode)
		:BevNodePrecondition(),
		m_exploration(exploration),
		m_leftProb(leftProb),
		m_rightProb(rightProb),
		m_isMaxWeightNode(isMaxWeightNode)
	{}

	virtual bool ExternalCondition(const BevNodeInputParam& input) const
	{
		double r = RandFloat();
		//s̰�����ԣ����������
		if (r > m_exploration)
		{
			//���ѡ��ڵ㣬�����ýڵ�����ֵ����
			r = RandFloat();
			if (r>=m_leftProb && r<m_rightProb)
			{
				return true;
			}
		}
		else
		{
			if (m_isMaxWeightNode)
			{
				return true;
			}
		}
		return false;
	}

	void setWeight(bool isMaxWeightNode){m_isMaxWeightNode = isMaxWeightNode;}
	void setPreprob(double leftProb, double rightProb){m_leftProb = leftProb; m_rightProb = rightProb;}

private:
	double m_exploration;
	double m_leftProb;
	double m_rightProb;
	bool  m_isMaxWeightNode;
};

class NOD_Flee : public BevNodeTerminal
{
public:
	NOD_Flee(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		pNearestEnemyPos = Vector2D(0,0);
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		Predator* pNearestEnemy = prey->GetSensorMem()->pNearestPredator;

		if (NULL != pNearestEnemy)
		{
			pNearestEnemyPos = pNearestEnemy->Pos();
			prey->GetSteering()->SetTarget(pNearestEnemyPos);
			prey->GetSteering()->FleeOn();
		}
	}

	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if (pNearestEnemyPos.isZero() || m_iCurExeTime>=m_iMaxExeTimes || prey->isDead())
		{
			prey->GetSteering()->FleeOff();
			return k_BRS_Finish;
		}
		return k_BRS_Executing;
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->FleeOff();
	}
private:
	Vector2D pNearestEnemyPos;
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

class NOD_SeekSafety : public BevNodeTerminal
{
public:
	NOD_SeekSafety(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		nearestHaven = NULL;
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		nearestHaven = prey->GetNearestHaven();

		if (NULL!=nearestHaven)
		{
			prey->GetSteering()->SetTarget(nearestHaven->Pos());
			prey->GetSteering()->SeekOn();
		}
	}
	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if (prey->InHavenZone() || NULL==nearestHaven  ||m_iCurExeTime>=m_iMaxExeTimes || prey->isDead())
		{
			prey->GetSteering()->SeekOff();
			return k_BRS_Finish;
		}
		return k_BRS_Executing;
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->SeekOff();
	}
private:
	BaseGameEntity* nearestHaven;
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};
class NOD_Forage : public BevNodeTerminal
{
public:
	NOD_Forage(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		nearestFood = NULL;
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		nearestFood = prey->GetNearestFood();

		if (NULL != nearestFood)
		{
			prey->GetSteering()->SetTarget(nearestFood->Pos());
			prey->GetSteering()->SeekOn();
		}
	}
	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if (prey->InFoodZone() || prey->isDead() || NULL==nearestFood || m_iCurExeTime>=m_iMaxExeTimes)
		{
			prey->GetSteering()->SeekOff();
			return k_BRS_Finish;
		}

		return k_BRS_Executing;
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->SeekOff();
	}
private:
	BaseGameEntity* nearestFood;
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

class NOD_Eat : public BevNodeTerminal
{
public:
	NOD_Eat(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 1;
	}
protected:
	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		if (prey->InFoodZone())
		{
			prey->GetWorld()->GetJudger()->PreyEatInterTable.push_back(prey);
		}

		return k_BRS_Finish;
	}
private:
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

class NOD_Flock : public BevNodeTerminal
{
public:
	NOD_Flock(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		prey->GetSteering()->FlockingOn();
	}
	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if(m_iCurExeTime >= m_iMaxExeTimes || prey->isDead())
		{
			prey->GetSteering()->FlockingOff();
			return k_BRS_Finish;
		}

		return k_BRS_Executing;
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->FlockingOff();
	}
private:
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

class NOD_Wander : public BevNodeTerminal
{
public:
	NOD_Wander(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		prey->GetSteering()->WanderOn();
	}
	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if(m_iCurExeTime >= m_iMaxExeTimes || prey->isDead())
		{
			prey->GetSteering()->WanderOff();
			return k_BRS_Finish;
		}
		return k_BRS_Executing;
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->WanderOff();
	}
private:
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

class NOD_Charge : public BevNodeTerminal
{
public:
	NOD_Charge(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		pNearestEnemy = NULL;
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;
		pNearestEnemy = prey->GetSensorMem()->pNearestPredator;

		if (NULL != pNearestEnemy)
		{
			prey->GetSteering()->PursuitOn(pNearestEnemy);
			prey->m_CurEnemy = pNearestEnemy;
		}
	}

	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if (NULL==pNearestEnemy || pNearestEnemy->isDead() || m_iCurExeTime>=m_iMaxExeTimes || prey->isDead())
		{
			prey->GetSteering()->PursuitOff();
			return k_BRS_Finish;
		}
		else
		{
			prey->GetWorld()->GetJudger()->PreytoPInterTable.insert(std::make_pair(prey,pNearestEnemy));
			//prey->GetWorld()->GetJudger()->JudgeAttackPreyToP(prey,pNearestEnemy);
			return k_BRS_Executing;
		}
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->PursuitOff();
	}
private:
	Predator* pNearestEnemy;
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

class NOD_Assist : public BevNodeTerminal
{
public:
	NOD_Assist(BevNode* _o_ParentNode)
		:BevNodeTerminal(_o_ParentNode)
	{
		m_iMaxExeTimes = 10;
	}
protected:
	virtual void _DoEnter(const BevNodeInputParam& input)
	{
		pAttackedEnemy = NULL;
		pAssistAlly = NULL;
		m_iCurExeTime = 0;

		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		std::vector<Agent*>::const_iterator curBot = prey->GetSensorMem()->m_MemoryMapPredators.begin();

		for (curBot; curBot!=prey->GetSensorMem()->m_MemoryMapPredators.end(); ++curBot)
		{
			pAssistAlly = dynamic_cast<Prey*>((*curBot)->GetSteering()->GetTargetAgent1());

			if (pAssistAlly!=NULL && pAssistAlly!=prey)
			{
				double dist = Vec2DDistanceSq(pAssistAlly->Pos(), prey->Pos());
				if(dist < prey->SensorRange()*prey->SensorRange())
				{
					pAttackedEnemy = dynamic_cast<Predator*>(*curBot);
					prey->GetSteering()->PursuitOn(pAttackedEnemy);
					prey->m_CurEnemy = pAttackedEnemy;
					break;
				}
			}
		}
	}

	virtual BevRunningStatus _DoExecute(const BevNodeInputParam& input, BevNodeOutputParam& output)	
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		m_iCurExeTime++;
		if (!pAttackedEnemy || !pAssistAlly || pAttackedEnemy->isDead() || pAssistAlly->isDead() || prey->isDead() || m_iCurExeTime>=m_iMaxExeTimes)
		{
			prey->GetSteering()->PursuitOff();
			prey->GetSteering()->SetTargetAgent1(NULL);
			return k_BRS_Finish;
		}
		else
		{
			prey->GetWorld()->GetJudger()->PreytoPInterTable.insert(std::make_pair(prey,pAttackedEnemy));
			//prey->GetWorld()->GetJudger()->JudgeAttackPreyToP(prey,pAttackedEnemy);
			return k_BRS_Executing;
		}
	}

	//reset default state for doTransition 
	virtual void _DoExit(const BevNodeInputParam& input, BevRunningStatus _ui_ExitID)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    prey = inputData.m_pOwner;

		prey->GetSteering()->PursuitOff();
		prey->GetSteering()->SetTargetAgent1(NULL);
	}
private:
	Predator* pAttackedEnemy;
	Prey* pAssistAlly;
	int m_iMaxExeTimes;
	int m_iCurExeTime;
};

//////////////////////////////////////////////////////////////////////////
//
//BevNodeLearningSelector
//
//////////////////////////////////////////////////////////////////////////
class BevNodeLearningSelector : public BevNodePrioritySelector
{
public:
	BevNodeLearningSelector(BevNode* _o_ParentNode, BevNodePrecondition* _o_NodePrecondition = NULL)
		: BevNodePrioritySelector(_o_ParentNode, _o_NodePrecondition)
	{
		child_Status = k_BRS_Finish;
	}

	virtual bool _DoEvaluate(const BevNodeInputParam& input)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Agent*    pA = inputData.m_pOwner;

		//��һ���ӽڵ�optionδ������ɣ�����ִ�У��������RLѡ���µĽڵ�
		if (child_Status)
		{
			m_pQlearner->updateEnvModelForRL(pA);
			//return action index other than action
			mui_CurrentSelectIndex = m_pQlearner->selectActionByRL(pA);
			m_pQlearner->RecordMemory(pA);
		}

		if(_bCheckIndex(mui_CurrentSelectIndex))
		{
			BevNode* oBN = mao_ChildNodeList[mui_CurrentSelectIndex];
			if(oBN->Evaluate(input))
			{
				return true;
			}
		}
		return false;
	}
	virtual BevRunningStatus _DoTick(const BevNodeInputParam& input, BevNodeOutputParam& output)
	{	
		child_Status = BevNodePrioritySelector::_DoTick(input,output);

		//in general, child node finish, then learning selector node finish
		myStatus = child_Status;

		if(child_Status)
		{
			//check whether satisfy the terminate conditions of current learning task
			if(!m_pQlearner->terminatedTask(m_pQlearner->pEnvModel->pState))
			{
				myStatus = k_BRS_Executing;
			}
		}

		return myStatus;
	}

	virtual void _DoNodeLearning(const BevNodeInputParam& input,BevNodeOutputParam& output)
	{
		const BevInputData&  inputData	= input.GetRealDataType<BevInputData>();
		Prey*    pA = inputData.m_pOwner;
		BevOutputData&  outData	= output.GetRealDataType<BevOutputData>();

		m_pQlearner->AccuRewardsOption(true,pA,outData.rewardPropogation);
		if (pA->isDead() || pA->InHavenZone())
		{
			child_Status = k_BRS_Finish;
		}

		//ѡ��ڵ����н����������ѧϰ֪ʶ
		if (child_Status)
		{
			m_pQlearner->KnowledgeLearn(pA);
			//m_bFire = false;
			pA->m_CurEnemy = NULL;
		}
	}

	bool isFire(){return m_bFire;}
	void setMyQlearner(QLearning* pQ){m_pQlearner = pQ;}

protected:
	BevRunningStatus child_Status;
	//bool m_bFire;
	QLearning* m_pQlearner;
};

