#include "GameWorld.h"
#include "Predator.h"
#include "Prey.h"
#include "constants.h"
#include "Obstacle.h"
#include "2d/Geometry.h"
#include "2d/Wall2D.h"
#include "2d/Transformations.h"
#include "SteeringBehaviors.h"
#include "time/PrecisionTimer.h"
#include "ParamLoader.h"
#include "misc/WindowUtils.h"
#include "misc/Stream_Utility_Functions.h"
#include "GameMap.h"
#include "Judger.h"
#include "AI/RL/QLearning.h"
#include "AI/RL/CPrimitiveAction.h"
#include "AI/RL/CAbstractionAction.h"


#include "resource.h"
#include <list>

//------------------------------- ctor -----------------------------------
//------------------------------------------------------------------------
GameWorld::GameWorld(int cx, int cy):

			m_cxClient(cx),
			m_cyClient(cy),
			m_bPaused(false),
			m_vCrosshair(Vector2D(cxClient()/2.0, cxClient()/2.0)),
			m_bShowWalls(false),
			m_bShowObstacles(false),
			m_bShowPath(false),
			m_bShowWanderCircle(false),
			m_bShowSteeringForce(false),
			m_bShowFeelers(false),
			m_bShowDetectionBox(false),
			m_bShowFPS(true),
			m_dAvFrameTime(0),
			m_pPath(NULL),
			m_bRenderNeighbors(false),
			m_bViewKeys(false),
			m_bShowCellSpaceInfo(false)
{

	//load in the default map
	m_pMap = new GameMap();
	m_pMap->LoadMap("PredatorPrey.map");

	//set judger for game——health increase of health bag, damage of predator vs prey,damage of prey vs predator, max tick num in each iterator
	m_pJudger = new Judger(Prm.DamagePreyToPredator,Prm.DamagePredatorToPrey,Prm.HealthGainInFoodZ,Prm.TimeOutGame,Prm.MaxRunTimes,this);

	ConstructQLearners();

	InitPredatorsList();
	InitPreysList();
}


//-------------------------------- dtor ----------------------------------
//------------------------------------------------------------------------
GameWorld::~GameWorld()
{
	ClearAgentsList();
  
	delete m_pPath;
	m_pPath = NULL;
	delete m_pMap;
	m_pMap = NULL;
	delete m_pJudger;
	m_pJudger = NULL;

	//for test, just list all simplly
	delete m_pRootQLearner;
	delete m_pRetreatQLearner;
	delete m_pFleeQLearner;
	delete m_pSeekSafetyQLearner;
	delete m_pIdleQLearner;
	delete m_pGrazeQLearner;
	delete m_pForageQLearner;
	delete m_pEatQLearner;
	delete m_pExploreQLearner;
	delete m_pFlockQLearner;
	delete m_pWanderQLearner;
	delete m_pAttackQLearner;
	delete m_pChargeQLearner;
	delete m_pAssistQLearner;
	m_pRootQLearner = NULL;
	m_pRetreatQLearner = NULL;
	m_pFleeQLearner = NULL;
	m_pSeekSafetyQLearner = NULL;
	m_pIdleQLearner = NULL;
	m_pGrazeQLearner = NULL;
	m_pForageQLearner = NULL;
	m_pEatQLearner = NULL;
	m_pExploreQLearner = NULL;
	m_pFlockQLearner = NULL;
	m_pWanderQLearner = NULL;
	m_pAttackQLearner = NULL;
	m_pChargeQLearner = NULL;
	m_pAssistQLearner = NULL;
}


//----------------------------- Update -----------------------------------
//------------------------------------------------------------------------
void GameWorld::Update(double time_elapsed)
{ 
//  if (m_bPaused) return;

  //create a smoother to smooth the framerate
  const int SampleRate = 10;
  static Smoother<double> FrameRateSmoother(SampleRate, 0.0);

  //m_dAvFrameTime = FrameRateSmoother.Update(time_elapsed);
  
  //update the predators
  for (unsigned int a=0; a<m_Predators.size(); ++a)
  {
	  if (m_Predators[a]->isAlive())
	  {
		  m_Predators[a]->Update(time_elapsed);
	  }
  }

  //update the preys
  for (unsigned int a=0; a<m_Preys.size(); ++a)
  {
	  if (m_pJudger->GetCurTicks()==0)
	  {
		  m_Preys[a]->UpdateSensorSys();
		  m_Preys[a]->UpdateAgentState();
	  }
	  //dead and inhaven are two absorbed states
	  if (m_Preys[a]->isAlive() && !m_Preys[a]->InHavenZone())
	  {
		  m_Preys[a]->SelectAction();
		  m_Preys[a]->m_bLearnNeeded = true;

		  m_Preys[a]->UpdateMovement(time_elapsed);
	  }
	  else
	  {
		  m_Preys[a]->m_bLearnNeeded = false;
	  }
  }

  //update new state after movement and interaction
  m_pJudger->CalculateCurDamage();
  for (unsigned int a=0; a<m_Preys.size(); ++a)
  {
	  m_Preys[a]->UpdateSensorSys();
	  m_Preys[a]->UpdateAgentState();
	  m_Preys[a]->SetDamage(false);
  }

  //preys learning
  for (unsigned int a=0; a<m_Preys.size(); ++a)
  {
	  if (m_Preys[a]->m_bLearnNeeded)
	  {
		  m_Preys[a]->ExecuteLearning();
	  }
  }

  //update episodes of game
  m_pJudger->AdvanceGameInEachRun();

  if(JudgeResult())
  {
	  //game over
	  if (m_pJudger->UptoMaxGameRuns())
	  {
		  //m_pGeneralQLearner->saveQTableToFile((char*)QTableFileAll.c_str());
		  m_pRootQLearner->saveQTableToFile((char*)QTableFileRoot.c_str());
		  m_pRetreatQLearner->saveQTableToFile((char*)QTableFileRetreat.c_str());
		  m_pAttackQLearner->saveQTableToFile((char*)QTableFileAttack.c_str());
		  m_pIdleQLearner->saveQTableToFile((char*)QTableFileIdle.c_str());

		  m_pJudger->SaveFinalStastics("result.txt");
		  //MessageBox(NULL, "GameOver!", "", MB_OK);
		  // kill the application, this sends a WM_QUIT message  
		  PostQuitMessage (0);
	  }
	  else//next run
	  {
		  ResetGame();
		  m_pJudger->AdvanceGameRuns();	
	  }
  }
}
  

//------------------------- Set Crosshair ------------------------------------
//
//  The user can set the position of the crosshair by right clicking the
//  mouse. This method makes sure the click is not inside any enabled
//  Obstacles and sets the position appropriately
//------------------------------------------------------------------------
void GameWorld::SetCrosshair(POINTS p)
{
  Vector2D ProposedPosition((double)p.x, (double)p.y);

  //make sure it's not inside an obstacle
  for (std::vector<BaseGameEntity*>::const_iterator curOb = GetGameMap()->GetObstacles().begin(); curOb != GetGameMap()->GetObstacles().end(); ++curOb)
  {
	if (PointInCircle((*curOb)->Pos(), (*curOb)->BRadius(), ProposedPosition))
	{
	  return;
	}

  }
  m_vCrosshair.x = (double)p.x;
  m_vCrosshair.y = (double)p.y;
}


//------------------------- HandleKeyPresses -----------------------------
void GameWorld::HandleKeyPresses(WPARAM wParam)
{

  //switch(wParam)
  //{
  //case 'U':
  //  {
  //    NULL = m_pPath;
  //    double border = 60;
  //    m_pPath = new Path(RandInt(3, 7), border, border, cxClient()-border, cyClient()-border, true); 
  //    m_bShowPath = true; 
  //    for (unsigned int i=0; i<m_Agents.size(); ++i)
  //    {
  //      m_Agents[i]->Steering()->SetPath(m_pPath->GetPath());
  //    }
  //  }
  //  break;

  //  case 'P':
  //    
  //    TogglePause(); break;

  //  case 'O':

  //    ToggleRenderNeighbors(); break;

  //  case 'I':

  //    {
  //      for (unsigned int i=0; i<m_Agents.size(); ++i)
  //      {
  //        m_Agents[i]->ToggleSmoothing();
  //      }

  //    }

  //    break;

  //  case 'Y':

  //     m_bShowObstacles = !m_bShowObstacles;

  //      if (!m_bShowObstacles)
  //      {
  //        m_Obstacles.clear();

  //        for (unsigned int i=0; i<m_Agents.size(); ++i)
  //        {
  //          m_Agents[i]->Steering()->ObstacleAvoidanceOff();
  //        }
  //      }
  //      else
  //      {
  //        CreateObstacles();

  //        for (unsigned int i=0; i<m_Agents.size(); ++i)
  //        {
  //          m_Agents[i]->Steering()->ObstacleAvoidanceOn();
  //        }
  //      }
  //      break;

  //}//end switch
}



//-------------------------- HandleMenuItems -----------------------------
void GameWorld::HandleMenuItems(WPARAM wParam, HWND hwnd)
{
  switch(wParam)
  {
	case ID_OB_OBSTACLES:

		//m_bShowObstacles = !m_bShowObstacles;

		//if (!m_bShowObstacles)
		//{
		//  m_Obstacles.clear();

		//  for (unsigned int i=0; i<m_Agents.size(); ++i)
		//  {
		//    m_Agents[i]->Steering()->ObstacleAvoidanceOff();
		//  }

		//  //uncheck the menu
		// ChangeMenuState(hwnd, ID_OB_OBSTACLES, MFS_UNCHECKED);
		//}
		//else
		//{
		//  CreateObstacles();

		//  for (unsigned int i=0; i<m_Agents.size(); ++i)
		//  {
		//    m_Agents[i]->Steering()->ObstacleAvoidanceOn();
		//  }

		//  //check the menu
		//  ChangeMenuState(hwnd, ID_OB_OBSTACLES, MFS_CHECKED);
		//}

	   break;

	case ID_OB_WALLS:

	 { //m_bShowWalls = !m_bShowWalls;

	  //if (m_bShowWalls)
	  //{
	  //  CreateWalls();

	  //  for (unsigned int i=0; i<m_Agents.size(); ++i)
	  //  {
	  //    m_Agents[i]->Steering()->WallAvoidanceOn();
	  //  }

	  //  //check the menu
	  //   ChangeMenuState(hwnd, ID_OB_WALLS, MFS_CHECKED);
	  //}

	  //else
	  //{
	  //  m_Walls.clear();

	  //  for (unsigned int i=0; i<m_Agents.size(); ++i)
	  //  {
	  //    m_Agents[i]->Steering()->WallAvoidanceOff();
	  //  }

	  //  //uncheck the menu
	  //   ChangeMenuState(hwnd, ID_OB_WALLS, MFS_UNCHECKED);
	  }

	  break;


	case IDR_PARTITIONING:
	  {
		//for (unsigned int i=0; i<m_Agents.size(); ++i)
		//{
		//  m_Agents[i]->Steering()->ToggleSpacePartitioningOnOff();
		//}

		////if toggled on, empty the cell space and then re-add all the 
		////Agents
		//if (m_Agents[0]->Steering()->isSpacePartitioningOn())
		//{
		//  m_pCellSpace->EmptyCells();
	   
		//  for (unsigned int i=0; i<m_Agents.size(); ++i)
		//  {
		//    m_pCellSpace->AddEntity(m_Agents[i]);
		//  }

		//  ChangeMenuState(hwnd, IDR_PARTITIONING, MFS_CHECKED);
		//}
		//else
		//{
		//  ChangeMenuState(hwnd, IDR_PARTITIONING, MFS_UNCHECKED);
		//  ChangeMenuState(hwnd, IDM_PARTITION_VIEW_NEIGHBORS, MFS_UNCHECKED);
		//  m_bShowCellSpaceInfo = false;

		//}
	  }

	  break;

	case IDM_PARTITION_VIEW_NEIGHBORS:
	  {
		//m_bShowCellSpaceInfo = !m_bShowCellSpaceInfo;
		//
		//if (m_bShowCellSpaceInfo)
		//{
		//  ChangeMenuState(hwnd, IDM_PARTITION_VIEW_NEIGHBORS, MFS_CHECKED);

		//  if (!m_Agents[0]->Steering()->isSpacePartitioningOn())
		//  {
		//    SendMessage(hwnd, WM_COMMAND, IDR_PARTITIONING, NULL);
		//  }
		//}
		//else
		//{
		//  ChangeMenuState(hwnd, IDM_PARTITION_VIEW_NEIGHBORS, MFS_UNCHECKED);
		//}
	  }
	  break;
		

	case IDR_WEIGHTED_SUM:
	  {
		ChangeMenuState(hwnd, IDR_WEIGHTED_SUM, MFS_CHECKED);
		ChangeMenuState(hwnd, IDR_PRIORITIZED, MFS_UNCHECKED);
		ChangeMenuState(hwnd, IDR_DITHERED, MFS_UNCHECKED);

		for (unsigned int i=0; i<m_Predators.size(); ++i)
		{
			m_Predators[i]->GetSteering()->SetSummingMethod(SteeringBehavior::weighted_average);
		}	
		for (unsigned int i=0; i<m_Preys.size(); ++i)
		{
			m_Preys[i]->GetSteering()->SetSummingMethod(SteeringBehavior::weighted_average);
		}
	  }

	  break;

	case IDR_PRIORITIZED:
	  {
		ChangeMenuState(hwnd, IDR_WEIGHTED_SUM, MFS_UNCHECKED);
		ChangeMenuState(hwnd, IDR_PRIORITIZED, MFS_CHECKED);
		ChangeMenuState(hwnd, IDR_DITHERED, MFS_UNCHECKED);

		for (unsigned int i=0; i<m_Predators.size(); ++i)
		{
		  m_Predators[i]->GetSteering()->SetSummingMethod(SteeringBehavior::prioritized);
		}
		for (unsigned int i=0; i<m_Preys.size(); ++i)
		{
			m_Preys[i]->GetSteering()->SetSummingMethod(SteeringBehavior::prioritized);
		}
	  }

	  break;

	case IDR_DITHERED:
	  {
		ChangeMenuState(hwnd, IDR_WEIGHTED_SUM, MFS_UNCHECKED);
		ChangeMenuState(hwnd, IDR_PRIORITIZED, MFS_UNCHECKED);
		ChangeMenuState(hwnd, IDR_DITHERED, MFS_CHECKED);

		for (unsigned int i=0; i<m_Predators.size(); ++i)
		{
		  m_Predators[i]->GetSteering()->SetSummingMethod(SteeringBehavior::dithered);
		}
		for (unsigned int i=0; i<m_Preys.size(); ++i)
		{
			m_Preys[i]->GetSteering()->SetSummingMethod(SteeringBehavior::dithered);
		}
	  }

	  break;


	  case ID_VIEW_KEYS:
	  //{
	  //  ToggleViewKeys();

	  //  CheckMenuItemAppropriately(hwnd, ID_VIEW_KEYS, m_bViewKeys);
	  //}

	  break;

	  case ID_VIEW_FPS:
	  {
		//ToggleShowFPS();

		//CheckMenuItemAppropriately(hwnd, ID_VIEW_FPS, RenderFPS());
	  }

	  break;

	  case ID_MENU_SMOOTHING:
	  {
		  //update the predators
		  for (unsigned int a=0; a<m_Predators.size(); ++a)
		  {
			  m_Predators[a]->ToggleSmoothing();
		  }
		  CheckMenuItemAppropriately(hwnd, ID_MENU_SMOOTHING, m_Predators[0]->isSmoothingOn());

		  //update the preys
		  for (unsigned int a=0; a<m_Preys.size(); ++a)
		  {
			  m_Preys[a]->ToggleSmoothing();
		  }

		CheckMenuItemAppropriately(hwnd, ID_MENU_SMOOTHING, m_Preys[0]->isSmoothingOn());
	  }

	  break;
	  
  }//end switch
}


//------------------------------ Render ----------------------------------
//------------------------------------------------------------------------
void GameWorld::Render()
{
  gdi->TransparentText();

  //render map
  m_pMap->Render();

  //render the predators
  for (unsigned int a=0; a<m_Predators.size(); ++a)
  {
	  m_Predators[a]->Render();  
  }  

  //render the preys
  for (unsigned int a=0; a<m_Preys.size(); ++a)
  {
	  m_Preys[a]->Render();  
  } 
}

//////////////////////////////Judge Game////////////////////////////////////////////
bool GameWorld::JudgeResult()
{
	GameStatus curStatus = m_pJudger->JudgeCurrentRunResult();

	if (curStatus > 0/* && curStatus != 3*/)
	{
		if (AllPreysInHaven == curStatus)
		{
			//MessageBox(NULL, "Prey Win: All preys are in safe haven!", "GameOver!", MB_OK);
		}
		else if (AllPreysDead == curStatus)
		{
			//MessageBox(NULL, "Predator Win: All preys are dead!", "GameOver!", MB_OK);
		}
		else if (AllPredatorsDead == curStatus)
		{
			//MessageBox(NULL, "Prey Win: All predators are dead!", "GameOver!", MB_OK);
			//////////////////////debug////////////////////////////////////////////////////
			//string outputAction = "Prey Win: All predators are dead!";

			//OutputDebugString(outputAction.c_str());
			///////////////////////////////////////////////////////////////////////////
		}
		else if (TimeOut == curStatus)
		{
			//MessageBox(NULL, "Prey Win: Time out!", "GameOver!", MB_OK);
		}
		return true;
	}

	return false;
}

void GameWorld::ResetGame()
{
	m_pJudger->ResetTicksInEachRun();
	ClearAgentsList();
	UpdateGLIEPolicy();

	InitPredatorsList();
	InitPreysList();
}

void GameWorld::ClearAgentsList()
{
	for (unsigned int a=0; a<m_Predators.size(); ++a)
	{
		delete m_Predators[a];
		m_Predators[a] = NULL;
	}
	m_Predators.clear();

	for (unsigned int a=0; a<m_Preys.size(); ++a)
	{
		delete m_Preys[a];
		m_Preys[a] = NULL;
	}
	m_Preys.clear();
}

void GameWorld::InitPredatorsList()
{
	//initialize predators
	for (int a=0; a<Prm.NumPredatorAgents; ++a)
	{
		//determine a  starting position
		Vector2D SpawnPos = Vector2D(RandFloat()*GetGameMap()->GetSizeX(),RandFloat()*GetGameMap()->GetSizeY());

		Predator* pPredator = new Predator(this,
			SpawnPos,                 //initial position
			RandFloat()*TwoPi,        //start rotation
			Vector2D(0,0),            //velocity
			Prm.AgentMass,          //mass
			Prm.MaxSteeringForce,     //max force
			Prm.MaxSpeed,             //max velocity
			Prm.MaxTurnRatePerSecond, //max turn rate
			Prm.AgentScale,			  //scale
			Prm.ViewDistance,
			Prm.MaxAgentHealth);        //viewdis

		//pPredator->GetSteering()->FlockingOn();
		pPredator->m_iAgentID = a;
		m_Predators.push_back(pPredator);
	}
}

void GameWorld::InitPreysList()
{
	//initialize preys
	for (int a=0; a<Prm.NumPreyAgents; ++a)
	{
		//determine a starting position
		Vector2D SpawnPos = Vector2D(RandFloat()*GetGameMap()->GetSizeX(),RandFloat()*GetGameMap()->GetSizeY());

		Prey* pPrey = new Prey(this,
			SpawnPos,                 //initial position
			RandFloat()*TwoPi,        //start rotation
			Vector2D(0,0),            //velocity
			Prm.AgentMass,          //mass
			Prm.MaxSteeringForce,     //max force
			Prm.MaxSpeed,             //max velocity
			Prm.MaxTurnRatePerSecond, //max turn rate
			Prm.AgentScale,			  //scale
			Prm.ViewDistance,
			Prm.MaxAgentHealth);        //viewdis

		//pPrey->GetSteering()->WanderOn();
		pPrey->m_iAgentID = a;
		pPrey->m_bLearnNeeded = true;
		m_Preys.push_back(pPrey);
	}
}

void GameWorld::ConstructQLearners()
{
	///////////////////////////定义多个学习器///////////////////////////////////////////////
	//分别为每个学习器构建响应的动作空间、状态空间动态生成 type区分每种符合动作的种类：选择1、序列2、并行3
	
	std::vector<CAction*> tempActionList;

	//1
	m_pRootQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pRootQLearner->LearnerName = "RootLearner";
	tempActionList.push_back(new CExtendedAction(1,EXTENDEDACTION::Retreat,0));
	tempActionList.push_back(new CExtendedAction(1,EXTENDEDACTION::Idle,1));
	tempActionList.push_back(new CExtendedAction(1,EXTENDEDACTION::Attack,2));
	m_pRootQLearner->setActionSet(tempActionList);
	tempActionList.clear();
	QTableFileRoot = "QTableRoot.txt";
	//m_pRootQLearner->loadQTableFromFile((char*)QTableFileRoot.c_str());

	//1-1
	m_pRetreatQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pRetreatQLearner->LearnerName = "RetreatLearner";
	//////分别为每个学习器构建响应的动作空间、状态空间
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Flee,0));
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::SeekSafety,1));
	m_pRetreatQLearner->setActionSet(tempActionList);
	tempActionList.clear();
	QTableFileRetreat = "QTableRetreat.txt";
	//m_pRetreatQLearner->loadQTableFromFile((char*)QTableFileRetreat.c_str());

	//1-1-1
	m_pFleeQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pFleeQLearner->LearnerName = "FleeLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Flee,0));
	m_pFleeQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-1-2
	m_pSeekSafetyQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pSeekSafetyQLearner->LearnerName = "SeeksafetyLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::SeekSafety,0));
	m_pSeekSafetyQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-2
	m_pIdleQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pIdleQLearner->LearnerName = "IdleLearner";
	tempActionList.push_back(new CExtendedAction(2,EXTENDEDACTION::Graze,0));
	tempActionList.push_back(new CExtendedAction(1,EXTENDEDACTION::Explore,1));
	m_pIdleQLearner->setActionSet(tempActionList);
	tempActionList.clear();
	QTableFileIdle = "QTableIdle.txt";
	//m_pIdleQLearner->loadQTableFromFile((char*)QTableFileIdle.c_str());

	//1-2-1
	m_pGrazeQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pGrazeQLearner->LearnerName = "GrazeQLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Forage,0));
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Eat,1));
	m_pGrazeQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-2-1-1
	m_pForageQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pForageQLearner->LearnerName = "ForageQLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Forage,0));
	m_pForageQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-2-1-2
	m_pEatQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pEatQLearner->LearnerName = "EatQLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Eat,0));
	m_pEatQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-2-2
	m_pExploreQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pExploreQLearner->LearnerName = "ExploreQLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Flock,0));
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Wander,1));
	m_pExploreQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-2-2-1
	m_pFlockQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pFlockQLearner->LearnerName = "FlockQLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Flock,0));
	m_pFlockQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-2-2-2
	m_pWanderQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pWanderQLearner->LearnerName = "WanderQLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Wander,0));
	m_pWanderQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-3
	m_pAttackQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pAttackQLearner->LearnerName = "AttackLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Charge,0));
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Assist,1));
	m_pAttackQLearner->setActionSet(tempActionList);
	tempActionList.clear();
	QTableFileAttack = "QTableAttack.txt";
	//m_pAttackQLearner->loadQTableFromFile((char*)QTableFileAttack.c_str());

	//1-3-1
	m_pChargeQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pChargeQLearner->LearnerName = "ChargeLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Charge,0));
	m_pChargeQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	//1-3-2
	m_pAssistQLearner = new QLearning(Prm.gamma,Prm.beta,Prm.exploration);
	m_pAssistQLearner->LearnerName = "AssistLearner";
	tempActionList.push_back(new PrimitiveAction(PRIMITIVEACTION::Assist,0));
	m_pAssistQLearner->setActionSet(tempActionList);
	tempActionList.clear();

	m_pRootQLearner->r_ChildrenLearners.insert(std::make_pair(m_pRetreatQLearner,0));
	m_pRootQLearner->r_ChildrenLearners.insert(std::make_pair(m_pIdleQLearner,1));
	m_pRootQLearner->r_ChildrenLearners.insert(std::make_pair(m_pAttackQLearner,2));
	m_pRetreatQLearner->r_ChildrenLearners.insert(std::make_pair(m_pFleeQLearner,0));
	m_pRetreatQLearner->r_ChildrenLearners.insert(std::make_pair(m_pSeekSafetyQLearner,1));
	m_pIdleQLearner->r_ChildrenLearners.insert(std::make_pair(m_pGrazeQLearner,0));
	m_pIdleQLearner->r_ChildrenLearners.insert(std::make_pair(m_pExploreQLearner,1));
	m_pGrazeQLearner->r_ChildrenLearners.insert(std::make_pair(m_pForageQLearner,0));
	m_pGrazeQLearner->r_ChildrenLearners.insert(std::make_pair(m_pEatQLearner,1));
	m_pExploreQLearner->r_ChildrenLearners.insert(std::make_pair(m_pFlockQLearner,0));
	m_pExploreQLearner->r_ChildrenLearners.insert(std::make_pair(m_pWanderQLearner,1));
	m_pAttackQLearner->r_ChildrenLearners.insert(std::make_pair(m_pChargeQLearner,0));
	m_pAttackQLearner->r_ChildrenLearners.insert(std::make_pair(m_pAssistQLearner,1));

	//inverse find
	m_pRootQLearner->ChildrenLearners.insert(std::make_pair(0,m_pRetreatQLearner));
	m_pRootQLearner->ChildrenLearners.insert(std::make_pair(1,m_pIdleQLearner));
	m_pRootQLearner->ChildrenLearners.insert(std::make_pair(2,m_pAttackQLearner));
	m_pRetreatQLearner->ChildrenLearners.insert(std::make_pair(0,m_pFleeQLearner));
	m_pRetreatQLearner->ChildrenLearners.insert(std::make_pair(1,m_pSeekSafetyQLearner));
	m_pIdleQLearner->ChildrenLearners.insert(std::make_pair(0,m_pGrazeQLearner));
	m_pIdleQLearner->ChildrenLearners.insert(std::make_pair(1,m_pExploreQLearner));
	m_pGrazeQLearner->ChildrenLearners.insert(std::make_pair(0,m_pForageQLearner));
	m_pGrazeQLearner->ChildrenLearners.insert(std::make_pair(1,m_pEatQLearner));
	m_pExploreQLearner->ChildrenLearners.insert(std::make_pair(0,m_pFlockQLearner));
	m_pExploreQLearner->ChildrenLearners.insert(std::make_pair(1,m_pWanderQLearner));
	m_pAttackQLearner->ChildrenLearners.insert(std::make_pair(0,m_pChargeQLearner));
	m_pAttackQLearner->ChildrenLearners.insert(std::make_pair(1,m_pAssistQLearner));

}


void GameWorld::UpdateGLIEPolicy()
{
	//learning rate decrease to 0 in limited maxrun;
	float decay = 0.995;
	float newbeta = m_pRootQLearner->beta*decay;
	float newexploration = m_pRootQLearner->exploration*decay;

	if(m_iMaxRunTimes-m_pJudger->m_iEpisode<100)
	{
		newbeta = 0;
		newexploration = 0;
	}

	m_pRootQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pRetreatQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pFleeQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pSeekSafetyQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pIdleQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pGrazeQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pForageQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pEatQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pExploreQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pFlockQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pWanderQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pAttackQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pChargeQLearner->setParameters(Prm.gamma,newbeta,newexploration);
	m_pAssistQLearner->setParameters(Prm.gamma,newbeta,newexploration);
}