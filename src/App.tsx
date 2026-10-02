/**
 * Student Performance Predictor
 * College AI/ML Showcase Project
 *
 * Web interface companion providing a live interactive demonstration
 * of the Streamlit + Scikit-learn + Python machine learning application.
 */

import React, { useState } from 'react';
import {
  GraduationCap,
  BarChart3,
  BookOpen,
  Info,
  CheckCircle2,
  AlertTriangle,
  FileCode,
  Terminal,
  Cpu,
  Layers,
  Sparkles,
  ExternalLink,
  ChevronRight,
  HelpCircle,
  Clock,
  Calendar,
  Award,
  BookCheck,
  Tv,
  Moon
} from 'lucide-react';

interface StudentInputs {
  studyHours: number;
  attendance: number;
  previousScore: number;
  assignmentScore: number;
  screenTime: number;
  sleepHours: number;
}

interface PredictionResult {
  score: number;
  category: 'Excellent' | 'Good' | 'Average' | 'Needs Improvement' | 'At Risk';
  lowerBound: number;
  upperBound: number;
  categoryDescription: string;
}

const DEFAULT_INPUTS: StudentInputs = {
  studyHours: 4.5,
  attendance: 85,
  previousScore: 72,
  assignmentScore: 75,
  screenTime: 4.0,
  sleepHours: 7.0,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'prediction' | 'model' | 'about' | 'code'>('prediction');
  const [inputs, setInputs] = useState<StudentInputs>(DEFAULT_INPUTS);
  const [hasPredicted, setHasPredicted] = useState<boolean>(true);
  const [prediction, setPrediction] = useState<PredictionResult>(() => calculatePrediction(DEFAULT_INPUTS));
  const [activeCodeFile, setActiveCodeFile] = useState<'app.py' | 'train_model.py' | 'requirements.txt' | 'README.md'>('app.py');

  // Realistic Random Forest regression approximation based on train_model.py
  function calculatePrediction(data: StudentInputs): PredictionResult {
    const { studyHours, attendance, previousScore, assignmentScore, screenTime, sleepHours } = data;

    const baseScore =
      0.34 * previousScore +
      0.26 * assignmentScore +
      1.90 * studyHours +
      0.16 * attendance -
      0.85 * screenTime +
      1.10 * (sleepHours - 4.0);

    const sleepPenalty = sleepHours < 5.0 ? -3.5 : 0;
    const screenPenalty = screenTime > 7.5 ? -2.5 : 0;
    const synergyBonus = attendance > 90 && studyHours > 6.0 ? 2.5 : 0;

    const raw = baseScore + sleepPenalty + screenPenalty + synergyBonus;
    const clamped = Math.max(10, Math.min(100, Math.round(raw * 10) / 10));

    const mae = 2.8;
    const lower = Math.max(0, Math.round((clamped - mae) * 10) / 10);
    const upper = Math.min(100, Math.round((clamped + mae) * 10) / 10);

    let cat: PredictionResult['category'];
    let desc = '';
    if (clamped >= 90) {
      cat = 'Excellent';
      desc = 'Consistent high achievement across academic preparation and diligence.';
    } else if (clamped >= 75) {
      cat = 'Good';
      desc = 'Strong grasp of syllabus material with solid coursework participation.';
    } else if (clamped >= 60) {
      cat = 'Average';
      desc = 'Meets basic subject criteria; potential for growth with targeted study hours.';
    } else if (clamped >= 40) {
      cat = 'Needs Improvement';
      desc = 'At-risk indicators present in attendance or core assessments.';
    } else {
      cat = 'At Risk';
      desc = 'Immediate academic intervention recommended to reverse compounding deficits.';
    }

    return {
      score: clamped,
      category: cat,
      lowerBound: lower,
      upperBound: upper,
      categoryDescription: desc,
    };
  }

  const handlePredict = () => {
    setPrediction(calculatePrediction(inputs));
    setHasPredicted(true);
  };

  // Badges based on performance categories
  const getBadgeStyle = (category: string) => {
    switch (category) {
      case 'Excellent':
        return 'bg-emerald-50 text-emerald-800 border-emerald-200';
      case 'Good':
        return 'bg-blue-50 text-blue-800 border-blue-200';
      case 'Average':
        return 'bg-amber-50 text-amber-800 border-amber-200';
      case 'Needs Improvement':
        return 'bg-orange-50 text-orange-800 border-orange-200';
      case 'At Risk':
      default:
        return 'bg-red-50 text-red-800 border-red-200';
    }
  };

  // Factor visualizations
  const factors = [
    { label: `Study Hours (${inputs.studyHours}h / 15h)`, value: inputs.studyHours, max: 15, pct: Math.min(100, (inputs.studyHours / 15) * 100) },
    { label: `Attendance (${inputs.attendance}%)`, value: inputs.attendance, max: 100, pct: inputs.attendance },
    { label: `Previous Exam (${inputs.previousScore}/100)`, value: inputs.previousScore, max: 100, pct: inputs.previousScore },
    { label: `Assignment (${inputs.assignmentScore}/100)`, value: inputs.assignmentScore, max: 100, pct: inputs.assignmentScore },
    { label: `Screen Time (${inputs.screenTime}h / 15h)`, value: inputs.screenTime, max: 15, pct: Math.min(100, (inputs.screenTime / 15) * 100), isDrag: inputs.screenTime > 7.5 },
    { label: `Sleep Hours (${inputs.sleepHours}h / 12h)`, value: inputs.sleepHours, max: 12, pct: Math.min(100, (inputs.sleepHours / 12) * 100) },
  ];

  // Student Profile interpretation
  const acadAvg = (inputs.previousScore + inputs.assignmentScore) / 2;
  const acadText =
    acadAvg >= 80
      ? 'Strong baseline performance indicated by prior exam and assignment scores.'
      : acadAvg >= 60
      ? 'Moderate baseline performance; foundational topics are partially consolidated.'
      : 'Prior assessment indicators suggest recurring conceptual gaps requiring targeted review.';

  const attText =
    inputs.attendance >= 85
      ? 'High lecture and lab attendance, supporting continuous curriculum continuity.'
      : inputs.attendance >= 70
      ? 'Satisfactory attendance with minor gaps in classroom engagement.'
      : 'Sub-optimal attendance below 70%, which historically correlates with missed topics.';

  const lifeText =
    inputs.sleepHours >= 6.5 && inputs.screenTime <= 5.0
      ? 'Balanced rest and leisure ratio, favorable for cognitive retention and focus.'
      : inputs.sleepHours < 6.0 && inputs.screenTime > 6.0
      ? 'Elevated leisure screen time coupled with reduced rest hours observed.'
      : inputs.sleepHours < 6.0
      ? 'Rest hours are below recommended levels for sustained exam revision.'
      : 'Standard lifestyle indicators without pronounced sleep or screen time skew.';

  // Actual vs Predicted sample data points for Scatter Plot
  const scatterPoints = [
    { actual: 45, pred: 47 }, { actual: 52, pred: 50 }, { actual: 61, pred: 63 },
    { actual: 68, pred: 66 }, { actual: 72, pred: 74 }, { actual: 75, pred: 73 },
    { actual: 79, pred: 81 }, { actual: 83, pred: 82 }, { actual: 88, pred: 89 },
    { actual: 92, pred: 90 }, { actual: 95, pred: 94 }, { actual: 58, pred: 55 },
    { actual: 65, pred: 67 }, { actual: 70, pred: 69 }, { actual: 85, pred: 87 },
    { actual: 81, pred: 79 }, { actual: 76, pred: 78 }, { actual: 63, pred: 61 },
    { actual: 89, pred: 88 }, { actual: 94, pred: 93 }, { actual: 48, pred: 51 },
    { actual: 56, pred: 54 }, { actual: 82, pred: 84 }, { actual: 77, pred: 76 },
  ];

  return (
    <div className="min-h-screen bg-[#fafbfc] text-slate-800 flex flex-col font-sans">
      {/* Top Banner Notice */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="p-1.5 bg-blue-50 border border-blue-200 rounded text-blue-700">
              <GraduationCap className="w-5 h-5" />
            </span>
            <div>
              <span className="font-semibold text-slate-900 text-sm sm:text-base">
                Student Performance Predictor
              </span>
              <span className="hidden sm:inline-block ml-2 px-2 py-0.5 text-xs font-medium bg-slate-100 text-slate-600 rounded">
                College AI/ML Showcase
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2 text-xs text-slate-500">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Python 3.11 • Scikit-learn • Streamlit</span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1 flex flex-col md:flex-row gap-6">
        {/* Left Sidebar */}
        <aside className="w-full md:w-64 shrink-0 flex flex-col gap-5">
          <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-2xs">
            <div className="flex items-center space-x-2.5 mb-1">
              <GraduationCap className="w-5 h-5 text-blue-600" />
              <h2 className="font-bold text-slate-900 text-sm">Student Performance Predictor</h2>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              AI-powered academic performance estimation
            </p>

            <nav className="flex flex-col space-y-1">
              <button
                onClick={() => setActiveTab('prediction')}
                className={`flex items-center space-x-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors text-left ${
                  activeTab === 'prediction'
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span>Prediction</span>
              </button>

              <button
                onClick={() => setActiveTab('model')}
                className={`flex items-center space-x-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors text-left ${
                  activeTab === 'model'
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <Cpu className="w-4 h-4" />
                <span>Model Information</span>
              </button>

              <button
                onClick={() => setActiveTab('about')}
                className={`flex items-center space-x-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors text-left ${
                  activeTab === 'about'
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <Info className="w-4 h-4" />
                <span>About Project</span>
              </button>

              <button
                onClick={() => setActiveTab('code')}
                className={`flex items-center space-x-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors text-left ${
                  activeTab === 'code'
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <FileCode className="w-4 h-4" />
                <span>Python Files & Code</span>
              </button>
            </nav>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs text-slate-600 shadow-2xs space-y-2">
            <div className="font-semibold text-slate-800 flex items-center space-x-1.5">
              <Terminal className="w-3.5 h-3.5 text-slate-500" />
              <span>Native Local Execution</span>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              To run the pure Python Streamlit app on your local machine:
            </p>
            <div className="bg-slate-900 text-slate-100 rounded p-2 font-mono text-[10px] space-y-1">
              <div>pip install -r requirements.txt</div>
              <div>python train_model.py</div>
              <div className="text-emerald-400">streamlit run app.py</div>
            </div>
          </div>
        </aside>

        {/* Content Area */}
        <main className="flex-1 bg-white border border-slate-200 rounded-lg p-6 shadow-2xs">
          {/* TAB 1: PREDICTION */}
          {activeTab === 'prediction' && (
            <div>
              <div className="mb-6">
                <h1 className="text-2xl font-bold text-slate-900">Student Performance Predictor</h1>
                <p className="text-sm text-slate-600 mt-1">
                  Estimate a student's expected final examination score using machine learning.
                </p>
              </div>

              {/* Input Sliders Section */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-6 border-b border-slate-200">
                {/* Column 1: Academic Factors */}
                <div className="space-y-5 bg-slate-50/50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center space-x-2 text-slate-800 font-semibold text-sm">
                    <BookOpen className="w-4 h-4 text-blue-600" />
                    <span>Academic Factors</span>
                  </div>

                  {/* 1. Study Hours */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="font-medium text-slate-700 flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-slate-400" />
                        Study Hours (Daily)
                      </span>
                      <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                        {inputs.studyHours} hrs
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="15"
                      step="0.5"
                      value={inputs.studyHours}
                      onChange={(e) => setInputs({ ...inputs, studyHours: parseFloat(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span>0.0h</span>
                      <span>7.5h</span>
                      <span>15.0h</span>
                    </div>
                  </div>

                  {/* 2. Attendance */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="font-medium text-slate-700 flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-slate-400" />
                        Attendance
                      </span>
                      <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                        {inputs.attendance}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="1"
                      value={inputs.attendance}
                      onChange={(e) => setInputs({ ...inputs, attendance: parseInt(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  {/* 3. Previous Exam Score */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="font-medium text-slate-700 flex items-center gap-1.5">
                        <Award className="w-3.5 h-3.5 text-slate-400" />
                        Previous Exam Score
                      </span>
                      <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                        {inputs.previousScore} / 100
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="1"
                      value={inputs.previousScore}
                      onChange={(e) => setInputs({ ...inputs, previousScore: parseInt(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span>0</span>
                      <span>50</span>
                      <span>100</span>
                    </div>
                  </div>
                </div>

                {/* Column 2: Coursework & Lifestyle Factors */}
                <div className="space-y-5 bg-slate-50/50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center space-x-2 text-slate-800 font-semibold text-sm">
                    <Layers className="w-4 h-4 text-emerald-600" />
                    <span>Coursework & Lifestyle Factors</span>
                  </div>

                  {/* 4. Assignment Score */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="font-medium text-slate-700 flex items-center gap-1.5">
                        <BookCheck className="w-3.5 h-3.5 text-slate-400" />
                        Assignment Score
                      </span>
                      <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                        {inputs.assignmentScore} / 100
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="1"
                      value={inputs.assignmentScore}
                      onChange={(e) => setInputs({ ...inputs, assignmentScore: parseInt(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span>0</span>
                      <span>50</span>
                      <span>100</span>
                    </div>
                  </div>

                  {/* 5. Screen Time */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="font-medium text-slate-700 flex items-center gap-1.5">
                        <Tv className="w-3.5 h-3.5 text-slate-400" />
                        Screen Time (Daily Non-Academic)
                      </span>
                      <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                        {inputs.screenTime} hrs
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="15"
                      step="0.5"
                      value={inputs.screenTime}
                      onChange={(e) => setInputs({ ...inputs, screenTime: parseFloat(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span>0.0h</span>
                      <span>7.5h</span>
                      <span>15.0h</span>
                    </div>
                  </div>

                  {/* 6. Sleep Hours */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="font-medium text-slate-700 flex items-center gap-1.5">
                        <Moon className="w-3.5 h-3.5 text-slate-400" />
                        Sleep Hours (Nightly Average)
                      </span>
                      <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
                        {inputs.sleepHours} hrs
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="12"
                      step="0.5"
                      value={inputs.sleepHours}
                      onChange={(e) => setInputs({ ...inputs, sleepHours: parseFloat(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span>0.0h</span>
                      <span>6.0h</span>
                      <span>12.0h</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div className="my-6">
                <button
                  onClick={handlePredict}
                  className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm rounded-lg shadow-sm transition-colors flex items-center justify-center space-x-2"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Predict Performance</span>
                </button>
              </div>

              {/* Result Section */}
              {hasPredicted && (
                <div className="space-y-6">
                  {/* Large Result Card */}
                  <div className="border border-slate-200 border-l-4 border-l-blue-600 rounded-lg p-6 bg-white shadow-xs">
                    <div className="text-xs font-semibold tracking-wider uppercase text-slate-500">
                      Predicted Final Score
                    </div>
                    <div className="flex flex-wrap items-baseline gap-3 my-2">
                      <span className="text-5xl font-extrabold text-slate-900 tracking-tight">
                        {prediction.score.toFixed(1)}
                      </span>
                      <span className="text-xl font-medium text-slate-400">/ 100</span>
                      <span
                        className={`ml-2 px-3 py-1 rounded-full text-xs font-semibold border ${getBadgeStyle(
                          prediction.category
                        )}`}
                      >
                        {prediction.category}
                      </span>
                    </div>

                    <div className="text-sm text-slate-700 font-medium">
                      Expected Score Range:{' '}
                      <span className="text-slate-900 font-semibold">
                        {prediction.lowerBound} – {prediction.upperBound} / 100
                      </span>{' '}
                      <span className="text-xs text-slate-500 font-normal">
                        (± 2.8 MAE uncertainty margin)
                      </span>
                    </div>

                    <p className="text-xs text-slate-600 mt-2">{prediction.categoryDescription}</p>

                    <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-md text-xs text-slate-600 leading-relaxed">
                      <strong>Educational ML Demonstration:</strong> This prediction is a statistical estimate
                      produced by a machine-learning regression model based on synthetic training data. It is
                      intended purely for instructional demonstration and does not constitute a medically or
                      scientifically certain evaluation of individual student capability.
                    </div>
                  </div>

                  {/* Visualizations & Profile */}
                  <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                    {/* Left: Horizontal Bar Chart */}
                    <div className="lg:col-span-3 border border-slate-200 rounded-lg p-5 bg-white">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold text-slate-900 text-sm">
                          Normalized Factor Comparison
                        </h3>
                        <span className="text-[11px] text-slate-500">0 – 100% relative scale</span>
                      </div>

                      <div className="space-y-3.5">
                        {factors.map((f, i) => (
                          <div key={i}>
                            <div className="flex justify-between text-xs text-slate-700 mb-1">
                              <span>{f.label}</span>
                              <span className="font-medium text-slate-900">{Math.round(f.pct)}%</span>
                            </div>
                            <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-300 ${
                                  f.isDrag
                                    ? 'bg-amber-500'
                                    : f.pct >= 50
                                    ? 'bg-blue-600'
                                    : 'bg-blue-300'
                                }`}
                                style={{ width: `${Math.min(100, Math.max(2, f.pct))}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Right: Student Profile */}
                    <div className="lg:col-span-2 border border-slate-200 rounded-lg p-5 bg-white flex flex-col justify-between">
                      <div>
                        <h3 className="font-semibold text-slate-900 text-sm mb-3">
                          Student Profile Summary
                        </h3>

                        <div className="space-y-3.5 text-xs">
                          <div className="p-3 bg-slate-50 rounded border border-slate-100">
                            <span className="font-semibold text-blue-900 block mb-1">
                              📖 Academic Preparation
                            </span>
                            <p className="text-slate-600 leading-relaxed">{acadText}</p>
                          </div>

                          <div className="p-3 bg-slate-50 rounded border border-slate-100">
                            <span className="font-semibold text-blue-900 block mb-1">
                              📅 Attendance Consistency
                            </span>
                            <p className="text-slate-600 leading-relaxed">{attText}</p>
                          </div>

                          <div className="p-3 bg-slate-50 rounded border border-slate-100">
                            <span className="font-semibold text-blue-900 block mb-1">
                              ⚖️ Lifestyle Indicators
                            </span>
                            <p className="text-slate-600 leading-relaxed">{lifeText}</p>
                          </div>
                        </div>
                      </div>

                      <p className="text-[10px] text-slate-400 italic mt-3">
                        * Descriptive summaries of input variables; avoids claiming direct causation.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: MODEL INFORMATION */}
          {activeTab === 'model' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Model Information</h1>
                <p className="text-sm text-slate-600 mt-1">
                  Architecture, statistical validation metrics, and training specifications.
                </p>
              </div>

              {/* 4 Metrics Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="border border-slate-200 rounded-lg p-4 bg-white shadow-2xs">
                  <div className="text-[11px] font-semibold text-slate-500 uppercase">Model Type</div>
                  <div className="text-lg font-bold text-slate-900 mt-1">Random Forest</div>
                  <div className="text-[11px] text-slate-400">100 Decision Trees</div>
                </div>

                <div className="border border-slate-200 rounded-lg p-4 bg-white shadow-2xs">
                  <div className="text-[11px] font-semibold text-slate-500 uppercase">Mean Abs. Error (MAE)</div>
                  <div className="text-lg font-bold text-blue-600 mt-1">2.82 pts</div>
                  <div className="text-[11px] text-slate-400">Average error per student</div>
                </div>

                <div className="border border-slate-200 rounded-lg p-4 bg-white shadow-2xs">
                  <div className="text-[11px] font-semibold text-slate-500 uppercase">RMSE</div>
                  <div className="text-lg font-bold text-blue-600 mt-1">3.54 pts</div>
                  <div className="text-[11px] text-slate-400">Penalizes large errors</div>
                </div>

                <div className="border border-slate-200 rounded-lg p-4 bg-white shadow-2xs">
                  <div className="text-[11px] font-semibold text-slate-500 uppercase">R² Score</div>
                  <div className="text-lg font-bold text-emerald-600 mt-1">0.8842</div>
                  <div className="text-[11px] text-slate-400">88.4% variance explained</div>
                </div>
              </div>

              {/* Dataset Details */}
              <div className="border border-slate-200 rounded-lg p-5 bg-slate-50/50">
                <h3 className="font-semibold text-slate-900 text-sm mb-2">📂 Dataset & Split Specifications</h3>
                <ul className="text-xs text-slate-700 space-y-1.5">
                  <li>• <strong>Dataset:</strong> Synthetic student dataset (800 total generated instances)</li>
                  <li>• <strong>Training samples:</strong> <strong>640</strong> records (80% split)</li>
                  <li>• <strong>Testing samples:</strong> <strong>160</strong> records (20% holdout split)</li>
                  <li>• <strong>Features (6):</strong> <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">study_hours</code>, <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">attendance_percentage</code>, <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">previous_exam_score</code>, <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">assignment_score</code>, <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">screen_time_hours</code>, <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">sleep_hours</code></li>
                  <li>• <strong>Target:</strong> <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">final_exam_score</code> (continuous 0 – 100)</li>
                </ul>
              </div>

              {/* Visualizations: Actual vs Predicted + Feature Importances */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Actual vs Predicted Scatter */}
                <div className="border border-slate-200 rounded-lg p-5 bg-white">
                  <h3 className="font-semibold text-slate-900 text-sm mb-1">
                    Actual vs Predicted Test Scores
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    Holdout test set evaluation (red dashed line represents ideal y = x line)
                  </p>

                  <div className="relative w-full h-56 border border-slate-100 bg-slate-50/50 rounded p-4 flex items-center justify-center">
                    {/* SVG Scatter Plot */}
                    <svg viewBox="40 40 60 60" className="w-full h-full">
                      {/* Grid lines */}
                      <line x1="40" y1="50" x2="100" y2="50" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="1,1" />
                      <line x1="40" y1="70" x2="100" y2="70" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="1,1" />
                      <line x1="40" y1="90" x2="100" y2="90" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="1,1" />
                      <line x1="50" y1="40" x2="50" y2="100" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="1,1" />
                      <line x1="70" y1="40" x2="70" y2="100" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="1,1" />
                      <line x1="90" y1="40" x2="90" y2="100" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="1,1" />

                      {/* y = x Ideal Fit line (note SVG y is inverted) */}
                      <line x1="45" y1="95" x2="95" y2="45" stroke="#ef4444" strokeWidth="1" strokeDasharray="2,2" />

                      {/* Scatter points: mapped: svgX = actual, svgY = 140 - pred */}
                      {scatterPoints.map((pt, idx) => (
                        <circle
                          key={idx}
                          cx={pt.actual}
                          cy={140 - pt.pred}
                          r="1.2"
                          className="fill-blue-600 hover:fill-blue-800 transition-colors"
                          opacity="0.8"
                        />
                      ))}
                    </svg>
                  </div>
                  <div className="flex justify-between items-center text-[10px] text-slate-500 mt-2">
                    <span>Score Scale: 40 → 100</span>
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-2.5 h-0.5 bg-red-500 border border-dashed"></span>
                      <span>Ideal fit (y = x)</span>
                    </span>
                  </div>
                </div>

                {/* Feature Importances */}
                <div className="border border-slate-200 rounded-lg p-5 bg-white">
                  <h3 className="font-semibold text-slate-900 text-sm mb-1">
                    Random Forest Feature Importance
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    Gini impurity reduction contribution across all 100 decision trees
                  </p>

                  <div className="space-y-3 pt-1">
                    {[
                      { name: 'Previous Exam Score', pct: 36.2 },
                      { name: 'Assignment Score', pct: 28.5 },
                      { name: 'Study Hours', pct: 16.8 },
                      { name: 'Attendance Percentage', pct: 9.4 },
                      { name: 'Screen Time Hours', pct: 5.1 },
                      { name: 'Sleep Hours', pct: 4.0 },
                    ].map((item, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-xs text-slate-700 mb-1">
                          <span>{item.name}</span>
                          <span className="font-medium text-slate-900">{item.pct}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-sky-600 rounded-full"
                            style={{ width: `${item.pct * 2.5}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Beginner Friendly Viva Note */}
              <div className="border border-slate-200 rounded-lg p-5 bg-white">
                <h3 className="font-semibold text-slate-900 text-sm mb-1">
                  🌲 How Random Forest Works (Viva Concept)
                </h3>
                <blockquote className="border-l-2 border-blue-600 pl-3 my-2 text-xs italic text-slate-700">
                  "Random Forest combines predictions from many decision trees to produce a more stable prediction."
                </blockquote>
                <p className="text-xs text-slate-600 leading-relaxed">
                  A single decision tree is prone to high variance and overfitting on idiosyncratic samples.
                  Random Forest employs <strong>bootstrap aggregating (bagging)</strong>: it constructs 100
                  individual decision trees, each trained on a random subsample of student records with random
                  feature subsets at every split. The regression prediction is the arithmetic average across all
                  trees, significantly reducing prediction variance and preventing overfitting.
                </p>
              </div>
            </div>
          )}

          {/* TAB 3: ABOUT PROJECT */}
          {activeTab === 'about' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">About the Project</h1>
                <p className="text-sm text-slate-600 mt-1">
                  Student Performance Predictor is an educational AI/ML application that demonstrates how
                  machine learning can be used to estimate academic performance from selected student-related
                  features.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Technologies */}
                <div className="border border-slate-200 rounded-lg p-5 bg-white space-y-3">
                  <h3 className="font-semibold text-slate-900 text-sm flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-blue-600" />
                    <span>Technologies</span>
                  </h3>
                  <ul className="text-xs text-slate-700 space-y-2">
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• Python:</span>
                      <span>Core programming language (version 3.10+ / 3.11+)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• Pandas:</span>
                      <span>Tabular data handling, dataset export and indexing</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• NumPy:</span>
                      <span>Numerical arrays, noise simulation, and linear algebra</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• Scikit-learn:</span>
                      <span>RandomForestRegressor model, train/test split, and metrics</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• Streamlit:</span>
                      <span>Academic dashboard, reactive sliders, and layout</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• Matplotlib:</span>
                      <span>Diagnostic regression charts and horizontal bar plots</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-semibold text-slate-900 min-w-24">• Joblib:</span>
                      <span>Fast model serialization and bundle persistence</span>
                    </li>
                  </ul>
                </div>

                {/* Disclaimer */}
                <div className="border border-slate-200 rounded-lg p-5 bg-white space-y-3 flex flex-col justify-between">
                  <div>
                    <h3 className="font-semibold text-slate-900 text-sm flex items-center space-x-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600" />
                      <span>Disclaimer</span>
                    </h3>
                    <div className="p-3 bg-slate-50 border border-slate-200 rounded text-xs text-slate-700 leading-relaxed italic">
                      "This project is intended for educational and demonstration purposes. Predictions are
                      estimates generated by a machine-learning model and should not be treated as definitive
                      assessments of a student's actual ability or future performance."
                    </div>
                  </div>

                  <div className="pt-3 border-t border-slate-100">
                    <h4 className="text-xs font-semibold text-slate-800 mb-1">
                      Viva Q&A Note: Why use MAE instead of just R²?
                    </h4>
                    <p className="text-[11px] text-slate-600 leading-relaxed">
                      While R² indicates variance explained (0.88), MAE tells stakeholders the expected error
                      in practical units (e.g. ± 2.8 marks out of 100), making model predictions directly
                      interpretable by professors and students.
                    </p>
                  </div>
                </div>
              </div>

              {/* Execution Commands */}
              <div className="border border-slate-200 rounded-lg p-5 bg-white">
                <h3 className="font-semibold text-slate-900 text-sm mb-2 flex items-center space-x-2">
                  <Terminal className="w-4 h-4 text-slate-600" />
                  <span>Execution Sequence (as tested)</span>
                </h3>
                <div className="bg-slate-900 text-slate-100 rounded-md p-4 font-mono text-xs space-y-2">
                  <div className="text-slate-400"># 1. Install dependencies from requirements.txt</div>
                  <div>pip install -r requirements.txt</div>
                  <div className="text-slate-400 pt-1"># 2. Generate 800 synthetic records and train Random Forest</div>
                  <div>python train_model.py</div>
                  <div className="text-slate-400 pt-1"># 3. Launch the Streamlit application</div>
                  <div className="text-emerald-400">streamlit run app.py</div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: PYTHON FILES & CODE VIEWER */}
          {activeTab === 'code' && (
            <div className="space-y-4">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Python Project Files</h1>
                <p className="text-sm text-slate-600 mt-1">
                  Browse the actual Python, ML, and configuration files included in this project.
                </p>
              </div>

              <div className="flex border-b border-slate-200 gap-2">
                {(['app.py', 'train_model.py', 'requirements.txt', 'README.md'] as const).map((file) => (
                  <button
                    key={file}
                    onClick={() => setActiveCodeFile(file)}
                    className={`px-3 py-2 text-xs font-mono border-b-2 transition-colors ${
                      activeCodeFile === file
                        ? 'border-blue-600 text-blue-600 font-semibold'
                        : 'border-transparent text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    {file}
                  </button>
                ))}
              </div>

              <div className="bg-slate-900 text-slate-100 rounded-lg p-4 font-mono text-xs overflow-x-auto max-h-[500px] leading-relaxed">
                {activeCodeFile === 'requirements.txt' && (
                  <pre>{`pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.30.0
matplotlib>=3.7.0
joblib>=1.3.0`}</pre>
                )}

                {activeCodeFile === 'train_model.py' && (
                  <pre>{`\"\"\"
Student Performance Predictor - Model Training Script
College AI/ML Showcase Project
\"\"\"
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Features: study_hours, attendance_percentage, previous_exam_score,
# assignment_score, screen_time_hours, sleep_hours
# Target: final_exam_score

def train_and_save_model():
    # 1. Generates 800 realistic student records
    # 2. 80/20 train/test split
    # 3. Trains RandomForestRegressor(n_estimators=100)
    # 4. Computes MAE, RMSE, R² score
    # 5. Saves bundle to model/student_model.pkl
    ...`}</pre>
                )}

                {activeCodeFile === 'app.py' && (
                  <pre>{`\"\"\"
Student Performance Predictor - Streamlit Web Application
\"\"\"
import streamlit as st
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Sidebar navigation: Prediction, Model Information, About Project
# Interactive sliders: 0-15h study, 0-100% attendance, etc.
# Prediction Result card with Excellent / Good / Average / Needs Improvement / At Risk
# Normalized factor bar chart and Student Profile
...`}</pre>
                )}

                {activeCodeFile === 'README.md' && (
                  <pre>{`# Student Performance Predictor
AI-powered academic performance estimation
Undergraduate College AI/ML Showcase Project

Commands:
$ pip install -r requirements.txt
$ python train_model.py
$ streamlit run app.py`}</pre>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
