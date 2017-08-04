#include "Prey.h"
#include "misc/cgdi.h"
#include "SensorMemory.h"
#include "SteeringBehaviors.h"
#include "2D/Vector2D.h"
#include "GameWorld.h"
#include "GameMap.h" 
#include "Zone.h"
#include "Judger.h"
#include "BaseGameEntity.h"

#include <iostream>
#include "BT/BevNodeFactory.h"
#include "AI/BT/BevTreeLocal.h"
#include "AI/RL/QLearning.h"
#include "AI/RL/CAbstractionAction.h"
#include "ModelBevLearning.h"
#include "AI/RL/EnvModel.h"
#include "AI/RL/Evaluator.h"
#include <map>


Prey::Prey(GameWorld* world,
	Vector2D position,
	double    rotation,
	Vector2D velocity,
	double    mass,
	double    max_force,
	double    max_speed,
	double    max_turn_rate,
	double    scale,
	double    ViewDistance,
	int       max_Health):		Agent(world,
									position,
									rotation,
									velocity,
									 mass,
									 max_force,
									 max_speed,
									 max_turn_rate,
									 scale,
									 ViewDistance,
									 max_Health)
{
	m_bInFoodZone = false;
	m_bInHavenZone = false;
	m_CurEnemy = NULL;
	pEvaluator = NULL;
	GetSteering()->WallAvoidanceOn();
	m_dReward = 0;

	CreateBTforAgent(NULL);
	BindLearnerForNode(NULL);
}


Prey::~Prey(void)
{
	D_SafeDelete(m_pBevTreeRoot);

	D_SafeDelete (pRootEnvModel);
	D_SafeDelete (pRetreatEnvModel);
	D_SafeDelete (pFleeEnvModel);
	D_SafeDelete (pSeekSafetyEnvModel);
	D_SafeDelete (pIdleEnvModel);
	D_SafeDelete (pGrazeEnvModel);
	D_SafeDelete (pForageEnvModel);
	D_SafeDelete (pEatEnvModel);
	D_SafeDelete (pExploreEnvModel);
	D_SafeDelete (pFlockEnvModel);
	D_SafeDelete (pWanderEnvModel);
	D_SafeDelete (pAttackEnvModel);
	D_SafeDelete (pChargeEnvModel);
	D_SafeDelete (pAssistEnvModel);

	D_SafeDelete(pEvaluator);
}


void Prey::SelectAction()
{
	/////////////////////////////////tick behavior tree/////////////////////////////////////////
	BevInputData m_BevTreeInputData;
	m_BevTreeInputData.m_pOwner = this;
	BevNodeInputParam input(&m_BevTreeInputData);
	BevOutputData m_BevTreeOutputData;
	BevNodeOutputParam output(&m_BevTreeOutputData);

	this->BindModelForLearner();

	if(m_pBevTreeRoot->Evaluate(input))
	{
		m_pBevTreeRoot->Tick(input, output);
	}

	BevNode* p = m_pBevTreeRoot->oGetLastActiveNode();

	//propogate to root
	while(p && p->GetParentNode())
	{
		//last time no active, selected just now 
		if(!p->m_bFire)
		{
			//record current state
			QLearning* pQ = NULL;
			if(NodeQMap.find(p) != NodeQMap.end())
			{
				pQ = NodeQMap.find(p)->second;
			}
			assert(pQ);

			pQ->pEnvModel->durationCount = 0;
			pQ->pEnvModel->startime = 0;
			pQ->pEnvModel->seq.clear();
			pQ->pEnvModel->seqDur.clear();
			
			QLearning* pQParent = NULL;
			if(NodeQMap.find(p->GetParentNode()) != NodeQMap.end())
			{
				pQParent = NodeQMap.find(p->GetParentNode())->second;
			}
			assert(pQParent);
			//record new state
			pQParent->updateEnvModelForRL(this);


			int actionIndex = -1;
			if(pQParent->r_ChildrenLearners.find(pQ) != pQParent->r_ChildrenLearners.end())
			{
				actionIndex = pQParent->r_ChildrenLearners.find(pQ)->second;
			}
			assert(actionIndex>=0);

			pQParent->selectActionByBT(this,actionIndex);
			pQParent->RecordMemory(this);

			if(p->GetNodeType() == CoreNodeType::k_NODE_ActionNode)
			{
				pQ->updateEnvModelForRL(this);
				pQ->selectActionByBT(this,0);
				pQ->RecordMemory(this);

				//push s onto the beginning of seq
				pQ->pEnvModel->seq.push_back(pQ->pEnvModel->preState);
			}
		}
		p = p->GetParentNode();
	}
}

void Prey::Render()
{
	//a vector to hold the transformed vertices
	static std::vector<Vector2D>  m_vecVehicleVBTrans;

	if (isAlive())
	{
		gdi->BluePen(); 
	}
	else if (isDead())
	{
		gdi->GreyPen();
	} 
	gdi->HollowBrush();

	m_vecVehicleVBTrans = WorldTransform(m_vecAgentVB,
		Pos(),
		Heading(),
		Side(),
		Scale());

	gdi->ClosedShape(m_vecVehicleVBTrans);
}

std::vector<Agent*>& Prey::GetAllyBotsInRange()
{
	return GetSensorMem()->m_MemoryMapPreys;
}


// override update(sensor,action,movment,state)
void Prey::Update(double time_elapsed)
{
	SelectAction();

	UpdateMovement(time_elapsed);

	UpdateSensorSys();

	UpdateAgentState();
}


// update if it is in Heaven zone
void Prey::UpdateInHeaven(void)
{
	std::vector<BaseGameEntity*>::const_iterator curHaven = GetWorld()->GetGameMap()->GetHavenZones().begin();
	for (curHaven; curHaven != GetWorld()->GetGameMap()->GetHavenZones().end(); ++curHaven)
	{
		double dist = Vec2DDistanceSq(Pos(),(*curHaven)->Pos());
		if (dist <= (-Scale().x+(*curHaven)->BRadius())*(-Scale().x+(*curHaven)->BRadius()))
		{
			setInHavenZone(true);
			break;
		}
		else
		{
			setInHavenZone(false);
		}
	}
}


// update if it is in Food Zone
void Prey::UpdateInFoodZone(void)
{
	std::vector<BaseGameEntity*>::const_iterator curFoodZone = GetWorld()->GetGameMap()->GetFoodZones().begin();
	for (curFoodZone; curFoodZone != GetWorld()->GetGameMap()->GetFoodZones().end(); ++curFoodZone)
	{
		double dist = Vec2DDistanceSq(Pos(),(*curFoodZone)->Pos());
		if (dist <= (Scale().x+(*curFoodZone)->BRadius())*(Scale().x+(*curFoodZone)->BRadius()))
		{
			setInFoodZone(true);
			break;
		}
		else
		{
			setInFoodZone(false);
		}
	}
}


// construct BT for agent
void Prey::CreateBTforAgent(const char* a_zXMLFile)
{
	//cannot construct BT from XML file
	if (a_zXMLFile == NULL)
	{
		std::cout << "xml file empty." << std::endl;
	}

	///////////////////////////////BT 分层控制///////////////////////////////////////////
	// 创建行为树的根节点
	m_pBevTreeRoot = GenerateLearningNode(NULL,"root"); 
	 
	m_pRetreat = GenerateLearningNode(m_pBevTreeRoot,"Retreat");
		m_pFlee = &BevNodeFactory::oCreateTeminalNode<NOD_Flee>(m_pRetreat,"Flee");
		m_pSeekSafety = &BevNodeFactory::oCreateTeminalNode<NOD_SeekSafety>(m_pRetreat,"SeekSafety");
	
	m_pIdle = GenerateLearningNode(m_pBevTreeRoot,"Idle");   // root-->Idle
		m_pGraze = &BevNodeFactory::oCreateSequenceNode(m_pIdle,"Graze");   // root-->Graze
			m_pForage = &BevNodeFactory::oCreateTeminalNode<NOD_Forage>(m_pGraze,"Forage");
			m_pEat = &BevNodeFactory::oCreateTeminalNode<NOD_Eat>(m_pGraze,"Eat");
		m_pExplore = &BevNodeFactory::oCreatePrioritySelectorNode(m_pIdle,"Explore");   // root-->Explore
		//m_pExplore = GenerateLearningNode(m_pIdle,"Explore");
			m_pFlock = &BevNodeFactory::oCreateTeminalNode<NOD_Flock>(m_pExplore,"Flock");
			m_pWander = &BevNodeFactory::oCreateTeminalNode<NOD_Wander>(m_pExplore,"Wander");

	m_pAttack = GenerateLearningNode(m_pBevTreeRoot,"Attack");
		m_pCharge = &BevNodeFactory::oCreateTeminalNode<NOD_Charge>(m_pAttack,"Charge");
		m_pAssist = &BevNodeFactory::oCreateTeminalNode<NOD_Assist>(m_pAttack,"Assist");
	
}

//构建学习选择节点
BevNode* Prey::GenerateLearningNode(BevNode* pParentNode,string NodeName)
{
	BevNodeLearningSelector* pReturn = new BevNodeLearningSelector(pParentNode);
	if (pParentNode)
	{
		pParentNode->AddChildNode(pReturn);
	}
	pReturn->SetDebugName(NodeName.c_str());
	return pReturn;
}

// update some states of agent recorded
void Prey::UpdateAgentState(void)
{
	UpdateInHeaven();
	UpdateInFoodZone();

	setNumAllyNeighbour(GetAllyBotsInRange().size());

	BaseGameEntity* nearestHaven = GetNearestHaven();
	if (nearestHaven)
	{
		setDisToNearestHaven(Vec2DDistance(Pos(),nearestHaven->Pos()));
	}
	else
	{
		setDisToNearestHaven(MaxInt);
	}

	BaseGameEntity* nearestFood = GetNearestFood();
	if (nearestFood)
	{
		setDisToNearestFood(Vec2DDistance(Pos(),nearestFood->Pos()));
	}
	else
	{
		setDisToNearestFood(MaxInt);
	}

	Predator* pNearestEnemy = GetNearestPredator();
	if (pNearestEnemy)
	{
		setDisToNearestPredator(Vec2DDistance(Pos(),pNearestEnemy->Pos()));
	}
	else
	{
		setDisToNearestPredator(MaxInt);
	}
}

void Prey::BindLearnerForNode( BevNode* pLearningNode )
{
	pEvaluator = new GeneralEvaluator();

	NodeQMap.insert(std::make_pair(m_pBevTreeRoot,GetWorld()->m_pRootQLearner));
	NodeQMap.insert(std::make_pair(m_pRetreat,GetWorld()->m_pRetreatQLearner));
	NodeQMap.insert(std::make_pair(m_pFlee,GetWorld()->m_pFleeQLearner));
	NodeQMap.insert(std::make_pair(m_pSeekSafety,GetWorld()->m_pSeekSafetyQLearner));
	NodeQMap.insert(std::make_pair(m_pIdle,GetWorld()->m_pIdleQLearner));
	NodeQMap.insert(std::make_pair(m_pGraze,GetWorld()->m_pGrazeQLearner));
	NodeQMap.insert(std::make_pair(m_pForage,GetWorld()->m_pForageQLearner));
	NodeQMap.insert(std::make_pair(m_pEat,GetWorld()->m_pEatQLearner));
	NodeQMap.insert(std::make_pair(m_pExplore,GetWorld()->m_pExploreQLearner));
	NodeQMap.insert(std::make_pair(m_pFlock,GetWorld()->m_pFlockQLearner));
	NodeQMap.insert(std::make_pair(m_pWander,GetWorld()->m_pWanderQLearner));
	NodeQMap.insert(std::make_pair(m_pAttack,GetWorld()->m_pAttackQLearner));
	NodeQMap.insert(std::make_pair(m_pCharge,GetWorld()->m_pChargeQLearner));
	NodeQMap.insert(std::make_pair(m_pAssist,GetWorld()->m_pAssistQLearner));

	//set qlearner for learning node to allow select action according to learning exploration policy
	dynamic_cast<BevNodeLearningSelector*>(m_pBevTreeRoot)->setMyQlearner(GetWorld()->m_pRootQLearner);
	dynamic_cast<BevNodeLearningSelector*>(m_pRetreat)->setMyQlearner(GetWorld()->m_pRetreatQLearner);
	dynamic_cast<BevNodeLearningSelector*>(m_pAttack)->setMyQlearner(GetWorld()->m_pAttackQLearner);
	dynamic_cast<BevNodeLearningSelector*>(m_pIdle)->setMyQlearner(GetWorld()->m_pIdleQLearner);
	
	//dynamic_cast<BevNodeLearningSelector*>(m_pExplore)->setMyQlearner(GetWorld()->m_pExploreQLearner);

	pRootEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pRetreatEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pFleeEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pSeekSafetyEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pIdleEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pGrazeEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pForageEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pEatEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pExploreEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pFlockEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pWanderEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pAttackEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pChargeEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
	pAssistEnvModel = new ModelSelGeneralBevLearning(this,pEvaluator);
}

void Prey::BindModelForLearner()
{
	GetWorld()->m_pRootQLearner->pEnvModel = pRootEnvModel;
	GetWorld()->m_pRetreatQLearner->pEnvModel = pRetreatEnvModel;
	GetWorld()->m_pFleeQLearner->pEnvModel = pFleeEnvModel; 
	GetWorld()->m_pSeekSafetyQLearner->pEnvModel = pSeekSafetyEnvModel;
	GetWorld()->m_pIdleQLearner->pEnvModel = pIdleEnvModel;
	GetWorld()->m_pGrazeQLearner->pEnvModel = pGrazeEnvModel;
	GetWorld()->m_pForageQLearner->pEnvModel =pForageEnvModel;
	GetWorld()->m_pEatQLearner->pEnvModel = pEatEnvModel;
	GetWorld()->m_pExploreQLearner->pEnvModel = pExploreEnvModel;
	GetWorld()->m_pFlockQLearner->pEnvModel = pFlockEnvModel;
	GetWorld()->m_pWanderQLearner->pEnvModel = pWanderEnvModel;
	GetWorld()->m_pAttackQLearner->pEnvModel = pAttackEnvModel;
	GetWorld()->m_pChargeQLearner->pEnvModel = pChargeEnvModel; 
	GetWorld()->m_pAssistQLearner->pEnvModel = pAssistEnvModel; 

}

void Prey::ExecuteLearning()
{
	this->BindModelForLearner();

	BevNode* p = m_pBevTreeRoot->oGetLastActiveNode();
	//died or finished after updating interaction, finish task
	if (this->isDead() || this->InHavenZone())
	{
		while(p)
		{
			p->myStatus = k_BRS_Finish;
			p = p->GetParentNode();
		}
	}
	p = m_pBevTreeRoot->oGetLastActiveNode();

	QLearning* pRootLearner = NodeQMap.find(this->m_pBevTreeRoot)->second;
	assert(pRootLearner);
	
	//propogate to root
	while(p && p->GetParentNode())
	{
		//child node finished, all state updating  
		if(p->myStatus)
		{
			QLearning* pQ = NULL;
			if(NodeQMap.find(p) != NodeQMap.end())
			{
				pQ = NodeQMap.find(p)->second;
			}
			assert(pQ);

			float rd = 0;
			if(p->GetNodeType() == CoreNodeType::k_NODE_ActionNode)//action node
			{
				pQ->pEnvModel->durationCount = pQ->pEnvModel->durationCount+1;
				pQ->AccuRewardsOption(true,this,pQ->pEnvModel->durationCount);

				//any action execution reward -1
				pQ->pEnvModel->rewardFeedback = pQ->pEnvModel->rewardFeedback - 1;

				pQ->UpdateVFunction(0,pQ->pEnvModel->preState,pQ->pEnvModel->rewardFeedback);
				rd = pQ->pEnvModel->rewardFeedback;
				//pQ->pEnvModel->rewardFeedback = 0;
				pQ->pEnvModel->seqDur.push_back(pQ->pEnvModel->durationCount);
			}

			if(p->GetNodeType() == CoreNodeType::k_NODE_ParalleNode)//
			{
				//pQ->RecordMemory(this);
				pQ->updateQValue(3,pQ->pEnvModel->preState,pQ->pEnvModel->preAction,rd,pQ->pEnvModel->pState,NULL,true,NULL);
			}

			if(p->GetParentNode())
			{
				QLearning* pQParent = NULL;
				if(NodeQMap.find(p->GetParentNode()) != NodeQMap.end())
				{
					pQParent = NodeQMap.find(p->GetParentNode())->second;
				}
				assert(pQParent);
				
				pQParent->AccuRewardsOption(false,this,pQ->pEnvModel->durationCount);
				
				if(pQParent->LearnerName != "RootLearner")//root node never be recalled by others, to save space no store
				{
					for(int i=0;i<pQ->pEnvModel->seq.size();i++)
					{
						pQParent->pEnvModel->seq.push_back(pQ->pEnvModel->seq[i]);
						pQParent->pEnvModel->seqDur.push_back(pQ->pEnvModel->seqDur[i]);
					}
				}

				//
				{
					if(p->GetParentNode()->GetNodeType()==CoreNodeType::k_NODE_SequenceNode || pQParent->LearnerName=="ExploreQLearner")
					{
						pQParent->updateQValue(2,pQParent->pEnvModel->preState,pQParent->pEnvModel->preAction,pQParent->pEnvModel->rewardFeedback,pQParent->pEnvModel->pState,NULL,p->myStatus,pQ);
					}
					else if(p->GetParentNode()->GetNodeType() == CoreNodeType::k_NODE_SelectorNode)
					{
						//option and MAXQ the finished flag different, option check if child is finished, in maxq we propogate if learner self is finished to lead hierarchical optimal  
						CState* mapState2 = pRootLearner->findStateinList(pQParent->pEnvModel->pState);
						float pesudoR = 0;
						if (p->GetParentNode()->myStatus) pesudoR = pRootLearner->getQValue(mapState2,pRootLearner->getBestQValueOnState(mapState2,true),true);
						
						pQParent->updateQValue(1,pQParent->pEnvModel->preState,pQParent->pEnvModel->preAction,pesudoR,pQParent->pEnvModel->pState,NULL,p->GetParentNode()->myStatus,pQ);
					}
				}
				
				pQParent->pEnvModel->durationCount = pQParent->pEnvModel->durationCount + pQParent->pEnvModel->preAction->getDuration();
			}

			p->m_bFire = false;
			this->m_CurEnemy = NULL;
		}
		else//executing
		{
			if(p->GetNodeType() == CoreNodeType::k_NODE_ActionNode)//primary action node
			{
				QLearning* pQ = NULL;
				if(NodeQMap.find(p) != NodeQMap.end())
				{
					pQ = NodeQMap.find(p)->second;
				}
				assert(pQ);

				pQ->pEnvModel->durationCount = pQ->pEnvModel->durationCount+1;
				pQ->AccuRewardsOption(true,this,pQ->pEnvModel->durationCount);
			}
			p->m_bFire = true;
		}
		p = p->GetParentNode();
	}
}