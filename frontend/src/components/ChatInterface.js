import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import '../ChatInterface.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:7071/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    try {
      const url = threadId
        ? `${API_URL}/chat/${threadId}`
        : `${API_URL}/chat`;

      const res = await axios.post(url, { message: text });

      if (res.data.threadId) {
        setThreadId(res.data.threadId);
      }

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.data.response },
      ]);
    } catch (err) {
      const errorMsg =
        err.response?.data?.error || err.message || 'Something went wrong';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `**Error:** ${errorMsg}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setThreadId(null);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Azure MCP Chat</h1>
        <button className="new-chat-btn" onClick={startNewChat}>
          New Chat
        </button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask me anything about your Azure resources.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-content">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-content loading-dots">
              <span>.</span><span>.</span><span>.</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
        }]);
      }
    } catch (error) {
      console.error('Error submitting travel request:', error);
      setMessages(prevMessages => [...prevMessages, { 
        role: 'bot', 
        content: 'Error submitting your travel request. Please try again.'
      }]);
      setFormSubmitted(false);
    } finally {
      setLoading(false);
    }
  };

  // Polling for status updates
  useEffect(() => {
    let intervalId;
    
    if (statusPolling && instanceId) {
      intervalId = setInterval(async () => {
        try {
          const statusResponse = await axios.get(`${API_URL}/travel-planner/status/${instanceId}`);
          const status = statusResponse.data;
          
          console.log("Status update received:", status); // Add logging to debug
          
          // Try to parse the custom status
          let customStatus = null;
          
          // Check for all possible ways the custom status might be provided
          if (status.customStatus) {
            customStatus = status.customStatus;
          } else if (status.CustomStatus) {
            customStatus = status.CustomStatus;
          } else if (status.serializedCustomStatus || status.SerializedCustomStatus) {
            // Handle serialized JSON string (needs parsing)
            try {
              const serialized = status.serializedCustomStatus || status.SerializedCustomStatus;
              customStatus = JSON.parse(serialized);
              console.log("Parsed serialized custom status:", customStatus);
            } catch (error) {
              console.error("Error parsing SerializedCustomStatus:", error);
            }
          }
          
          // Update the orchestration status if we found custom status info
          if (customStatus) {
            setOrchestrationStatus(customStatus);
            
            // Check if we're at the waiting for approval step
            if (customStatus && customStatus.step === "WaitingForApproval") {
              setLoading(false);
              setPlanReadyForApproval(true);
              setApprovalStatus("waiting");
              
              // If there is plan data in the custom status, use it
              if (customStatus.travelPlan) {
                console.log("Found travel plan in custom status:", customStatus.travelPlan);
                console.log("Document URL in custom status:", customStatus.documentUrl);
                
                // Create a complete plan object from the custom status
                const completePlan = {
                  Plan: {
                    itinerary: {
                      destinationName: customStatus.destination,
                      travelDates: customStatus.travelPlan.dates,
                      estimatedTotalCost: customStatus.travelPlan.cost,
                      dailyPlan: customStatus.travelPlan.dailyPlan || [] // Include the full dailyPlan
                    },
                    attractions: customStatus.travelPlan.attractions || [], // Include the attractions
                    restaurants: customStatus.travelPlan.restaurants || [], // Include the restaurants
                    insiderTips: customStatus.travelPlan.insiderTips || "No insider tips available", // Include the insiderTips
                    documentUrl: customStatus.documentUrl
                  },
                  documentUrl: customStatus.documentUrl // Add it at root level too
                };
                
                setPlanData(completePlan);
                displayTravelPlanForApproval(completePlan);
              }
            }
            // Check if we're in the booking step
            else if (customStatus && customStatus.step === "BookingTrip") {
              setLoading(true);
              setPlanReadyForApproval(false);
              setApprovalStatus("processing");
              
              // Add booking in progress message
              setMessages(prevMessages => {
                const hasBookingMessage = prevMessages.some(msg => 
                  msg.role === 'bot' && msg.content.includes("Booking your trip"));
                
                if (!hasBookingMessage) {
                  return [...prevMessages, { 
                    role: 'bot', 
                    content: `🎯 **Booking your trip to ${customStatus.destination}...**\n\nPlease wait while we confirm your reservation details.`
                  }];
                }
                return prevMessages;
              });
            }
            // Check if we're completed with booking
            else if (customStatus && customStatus.step === "Completed") {
              setStatusPolling(false);
              setLoading(false);
              setPlanReadyForApproval(false);
              setApprovalStatus("approved");
              
              // Add booking success message
              setMessages(prevMessages => {
                const hasCompletedMessage = prevMessages.some(msg => 
                  msg.role === 'bot' && msg.content.includes("Trip Successfully Booked"));
                
                if (!hasCompletedMessage) {
                  return [...prevMessages, { 
                    role: 'bot', 
                    content: `# 🎉 Trip Successfully Booked!\n\n**Destination:** ${customStatus.destination}\n**Booking ID:** ${customStatus.booking_id || 'Generating...'}\n\nYour trip has been confirmed! Check your email for detailed confirmation and itinerary documents.`
                  }];
                }
                return prevMessages;
              });
            }
          }
          // Check if the orchestration is completed
          else if (status.runtimeStatus === 'Completed' || status.RuntimeStatus === 'Completed') {
            setStatusPolling(false);
            setLoading(false);
            setPlanReadyForApproval(false);
            
            // Add the result to the chat
            if (status.output || status.Output) {
              const plan = status.output || status.Output;
              setPlanData(plan);
              
              // Check if this is a completed booking or just a plan ready for approval
              if (plan.BookingResult || plan.bookingResult) {
                // This is a completed booking
                setApprovalStatus("approved");
                const bookingResult = plan.BookingResult || plan.bookingResult;
                const bookingId = bookingResult.booking_id || bookingResult.bookingId || 'N/A';
                const destination = bookingResult.destination || 'Your destination';
                
                setMessages(prevMessages => {
                  const hasCompletedMessage = prevMessages.some(msg => 
                    msg.role === 'bot' && msg.content.includes("Trip Successfully Booked"));
                  
                  if (!hasCompletedMessage) {
                    return [...prevMessages, { 
                      role: 'bot', 
                      content: `# 🎉 Trip Successfully Booked!\n\n**Destination:** ${destination}\n**Booking ID:** ${bookingId}\n\nYour trip has been confirmed! ${bookingResult.message || 'Check your email for detailed confirmation.'}`
                    }];
                  }
                  return prevMessages;
                });
              } else if (plan.bookingConfirmation && plan.bookingConfirmation.includes("Booking confirmed")) {
                setApprovalStatus("approved");
                displayBookingConfirmation(plan);
              } else if (plan.bookingConfirmation && plan.bookingConfirmation.includes("not approved")) {
                setApprovalStatus("rejected");
                displayRejectionMessage(plan);
              } else {
                // Show the travel plan for approval
                setPlanReadyForApproval(true);
                setApprovalStatus("waiting");
                displayTravelPlanForApproval(plan);
              }
            }
          } else if (status.runtimeStatus === 'Failed' || status.RuntimeStatus === 'Failed') {
            setStatusPolling(false);
            setLoading(false);
            setMessages(prevMessages => [...prevMessages, { 
              role: 'bot', 
              content: `Unfortunately, there was an error processing your travel plan. Please try again.`
            }]);
          }
        } catch (error) {
          console.error('Error checking status:', error);
        }
      }, 5000); // Check every 5 seconds
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [statusPolling, instanceId]);

  // Effect to check for trip confirmation status after approval
  useEffect(() => {
    let confirmationIntervalId;
    
    if (instanceId && approvalStatus === "processing") {
      confirmationIntervalId = setInterval(async () => {
        try {
          // Reuse the status endpoint to check for completion
          const statusResponse = await axios.get(`${API_URL}/travel-planner/status/${instanceId}`);
          const status = statusResponse.data;
          
          console.log("Confirmation check - status received:", status);
          
          // Check if orchestration is completed
          if (status.runtimeStatus === "Completed" || status.RuntimeStatus === "Completed") {
            const output = status.output || status.Output || {};
            
            // Check for booking result (trip was approved and booked)
            const bookingResult = output.BookingResult || output.bookingResult;
            const bookingConfirmation = output.BookingConfirmation || output.bookingConfirmation || "";
            
            if (bookingResult && (bookingResult.status === "confirmed" || bookingResult.booking_id)) {
              // Trip is confirmed
              setConfirmationStatus("confirmed");
              setApprovalStatus("approved");
              setPlanReadyForApproval(false);
              clearInterval(confirmationIntervalId);
              
              // Update the UI with confirmation message
              const updatedPlanData = planData ? { ...planData, bookingConfirmation: bookingConfirmation, BookingResult: bookingResult } : null;
              if (updatedPlanData) {
                displayBookingConfirmation(updatedPlanData);
              }
            } else if (bookingConfirmation.toLowerCase().includes("not approved") || bookingConfirmation.toLowerCase().includes("timed out")) {
              // Trip was rejected
              setConfirmationStatus("rejected");
              setApprovalStatus("rejected");
              setPlanReadyForApproval(false);
              clearInterval(confirmationIntervalId);
              
              // Update UI with rejection message
              const updatedPlanData = planData ? { ...planData, bookingConfirmation: bookingConfirmation } : null;
              if (updatedPlanData) {
                displayRejectionMessage(updatedPlanData);
              }
            } else {
              // Completed but unclear - stop polling
              clearInterval(confirmationIntervalId);
              setLoading(false);
            }
          } else if (status.runtimeStatus === "Failed" || status.RuntimeStatus === "Failed") {
            clearInterval(confirmationIntervalId);
            setLoading(false);
            setApprovalStatus("failed");
          }
        } catch (error) {
          console.error('Error checking confirmation status:', error);
        }
      }, 3000); // Check every 3 seconds
    }
    
    return () => {
      if (confirmationIntervalId) clearInterval(confirmationIntervalId);
    };
  }, [instanceId, approvalStatus, planData]);

  // Helper function to display travel plan for approval
  const displayTravelPlanForApproval = (plan) => {
    console.log("Displaying plan for approval:", plan);
    
    // Normalize the data structure regardless of format
    let destinationName, travelDates, dailyPlan = [], estimatedTotalCost;
    let attractions = [], restaurants = [], insiderTips = "No insider tips available";
    let recommendations = [];
    
    // Check if we have a direct agent response or a wrapped response
    if (plan.DestinationName || plan.Attractions || plan.dailyPlan || plan.DailyPlan) {
      // Direct agent response format
      console.log("Processing direct agent response format");
      
      destinationName = plan.DestinationName || plan.destinationName || "Your destination";
      travelDates = plan.TravelDates || plan.travelDates || "Your travel dates";
      dailyPlan = plan.DailyPlan || plan.dailyPlan || [];
      estimatedTotalCost = plan.EstimatedTotalCost || plan.estimatedTotalCost || "Cost not available";
      attractions = plan.Attractions || plan.attractions || [];
      restaurants = plan.Restaurants || plan.restaurants || [];
      insiderTips = plan.InsiderTips || plan.insiderTips || insiderTips;
    } else {
      // Standard wrapped response
      const planData = plan.Plan || plan.plan || {};
      
      if (!planData) {
        console.error("No plan data found in:", plan);
        return;
      }
      
      // Get recommendations
      recommendations = planData.DestinationRecommendations?.Recommendations || 
                       planData.destinationRecommendations?.recommendations || [];
      
      // Get itinerary data
      const itinerary = planData.Itinerary || planData.itinerary || {};
      
      destinationName = itinerary?.DestinationName || itinerary?.destinationName || 
                       planData?.DestinationName || planData?.destinationName || "Your destination";
      
      travelDates = itinerary?.TravelDates || itinerary?.travelDates || 
                   planData?.TravelDates || planData?.travelDates || "Your travel dates";
      
      // Find daily plan in all possible locations
      if (itinerary?.DailyPlan && itinerary.DailyPlan.length > 0) {
        dailyPlan = itinerary.DailyPlan;
      } else if (itinerary?.dailyPlan && itinerary.dailyPlan.length > 0) {
        dailyPlan = itinerary.dailyPlan;
      } else if (planData?.DailyPlan && planData.DailyPlan.length > 0) {
        dailyPlan = planData.DailyPlan;
      } else if (planData?.dailyPlan && planData.dailyPlan.length > 0) {
        dailyPlan = planData.dailyPlan;
      }
      
      estimatedTotalCost = itinerary?.EstimatedTotalCost || itinerary?.estimatedTotalCost || 
                          planData?.EstimatedTotalCost || planData?.estimatedTotalCost || "Cost not available";
      
      // Get local recommendations
      const localRecommendations = planData.LocalRecommendations || planData.localRecommendations || {};
      insiderTips = localRecommendations?.InsiderTips || localRecommendations?.insiderTips || 
                   planData?.InsiderTips || planData?.insiderTips || insiderTips;
      
      // Get attractions and restaurants
      attractions = planData.Attractions || planData.attractions || [];
      restaurants = planData.Restaurants || planData.restaurants || [];
    }
    
    console.log(`Destination: ${destinationName}, Days: ${dailyPlan.length}, Attractions: ${attractions.length}`);
    
    // Format attractions and restaurants into readable content
    let localRecommendationsContent = "";
    
    // Format attractions
    if (attractions.length > 0) {
      localRecommendationsContent += "### Must-Visit Attractions\n";
      attractions.forEach(attraction => {
        const name = attraction.Name || attraction.name;
        const category = attraction.Category || attraction.category || "Attraction";
        const description = attraction.Description || attraction.description || "";
        const location = attraction.Location || attraction.location || "Destination area";
        const cost = attraction.EstimatedCost || attraction.estimatedCost || "Varies";
        const rating = attraction.Rating || attraction.rating || 4.0;
        
        localRecommendationsContent += `* **${name}** (${category}) - ${rating}⭐\n  ${description}\n  _Located in ${location}, Cost: ${cost}_\n\n`;
      });
    }
    
    // Format restaurants
    if (restaurants.length > 0) {
      localRecommendationsContent += "### Recommended Restaurants\n";
      restaurants.forEach(restaurant => {
        const name = restaurant.Name || restaurant.name;
        const cuisine = restaurant.Cuisine || restaurant.cuisine || "Various cuisines";
        const description = restaurant.Description || restaurant.description || "";
        const location = restaurant.Location || restaurant.location || "Destination area";
        const priceRange = restaurant.PriceRange || restaurant.priceRange || "$$";
        const rating = restaurant.Rating || restaurant.rating || 4.0;
        
        localRecommendationsContent += `* **${name}** (${cuisine}) - ${rating}⭐\n  ${description}\n  _Located in ${location}, Price: ${priceRange}_\n\n`;
      });
    }
    
    // Add text-based insider tips if available
    if (insiderTips && insiderTips !== "No insider tips available") {
      localRecommendationsContent += "### Insider Tips\n" + insiderTips;
    }
    
    // If we have no local recommendations content but have insider tips
    if (localRecommendationsContent === "" && insiderTips) {
      localRecommendationsContent = insiderTips;
    }
    
    // Add notification message first
    setMessages(prevMessages => {
      // Avoid duplicates
      if (!prevMessages.some(msg => msg.role === 'bot' && msg.content.includes("Your travel plan is now ready"))) {
        return [...prevMessages, { 
          role: 'bot', 
          content: `## Your travel plan is now ready for your review!\n\nPlease review the details below and decide if you'd like to proceed with this plan.` 
        }];
      }
      return prevMessages;
    });
    
    // Format the full travel plan content
    let resultContent = `
# Your Travel Plan is Ready for Review!

## Destination Recommendation
${recommendations.length > 0 
  ? recommendations.map(rec => {
      const name = rec.DestinationName || rec.destinationName;
      const description = rec.Description || rec.description || "";
      const matchScore = rec.MatchScore || rec.matchScore || "N/A";
      return `* **${name}** (Match: ${matchScore}%)\n  ${description}`;
    }).join('\n')
  : `* **${destinationName}**\n  Selected based on your preferences.`
}

## Itinerary Highlights
**${destinationName} - ${travelDates}**

${dailyPlan.length > 0 
  ? dailyPlan.map(day => {
      const dayNum = day.Day || day.day;
      const date = day.Date || day.date;
      const activities = day.Activities || day.activities || [];
      
      return `### Day ${dayNum} - ${date}\n${activities.map(act => {
        const time = act.Time || act.time;
        const name = act.ActivityName || act.activityName;
        const location = act.Location || act.location;
        const cost = act.EstimatedCost || act.estimatedCost;
        
        return `* ${time}: ${name} at ${location} (${cost})`;
      }).join('\n')}`;
    }).join('\n\n')
  : "Daily itinerary being finalized."
}

## Local Recommendations
${localRecommendationsContent || "Local recommendations being prepared."}

## Total Estimated Cost
${estimatedTotalCost}

## Next Steps
Please review this travel plan and click "Yes, Book My Trip!" below if you'd like to proceed with booking, or "No, I Need Changes" if you'd like to make modifications.
    `;
    
    // Add the detailed plan to the messages
    setMessages(prevMessages => {
      // Avoid duplicates
      if (!prevMessages.some(msg => msg.role === 'bot' && msg.content.includes("Your Travel Plan is Ready for Review"))) {
        return [...prevMessages, { role: 'bot', content: resultContent }];
      }
      return prevMessages;
    });
  };

  // Helper function to display booking confirmation
  const displayBookingConfirmation = (plan) => {
    // Handle both lowercase and uppercase property names
    console.log("Displaying booking confirmation for plan:", plan);
    
    // Safely access nested properties with fallbacks for different casing
    const planData = plan.Plan || plan.plan;
    
    if (!planData) {
      console.error("No plan data found in:", plan);
      return;
    }
    
    const itinerary = planData.Itinerary || planData.itinerary;
    
    if (!itinerary) {
      console.error("No itinerary found in plan data:", planData);
      return;
    }
    
    const destinationName = itinerary.DestinationName || itinerary.destinationName || "Your destination";
    const travelDates = itinerary.TravelDates || itinerary.travelDates || "Your travel dates";
    
    // Get booking confirmation text
    const bookingConfirmation = plan.BookingConfirmation || plan.bookingConfirmation || "Your booking has been confirmed.";
    
    // Get document URL - check all possible locations where it might be nested
    const documentUrl = plan.DocumentUrl || plan.documentUrl || 
                       planData.DocumentUrl || planData.documentUrl || 
                       itinerary.DocumentUrl || itinerary.documentUrl || 
                       "No document URL available";
    
    console.log("Document URL found:", documentUrl);
    
    let resultContent = `
# Your Trip Has Been Booked!

## Booking Confirmation
${bookingConfirmation}

## Travel Plan Details
Destination: ${destinationName}
Dates: ${travelDates}

## Document URL
${documentUrl}
    `;
    
    setMessages(prevMessages => {
      // Check if we already have this message to avoid duplicates
      const isDuplicate = prevMessages.some(msg => 
        msg.role === 'bot' && msg.content.includes("Your Trip Has Been Booked!"));
      
      if (!isDuplicate) {
        return [...prevMessages, { role: 'bot', content: resultContent }];
      }
      return prevMessages;
    });
  };

  // Helper function to display rejection message
  const displayRejectionMessage = (plan) => {
    let resultContent = `
# Travel Plan Rejected

Your travel plan was not approved. You can start a new travel plan when you're ready.

## Comments
${plan.bookingConfirmation.replace("Travel plan was not approved. Comments: ", "")}
    `;
    
    setMessages(prevMessages => {
      // Check if we already have this message to avoid duplicates
      const isDuplicate = prevMessages.some(msg => 
        msg.role === 'bot' && msg.content.includes("Travel Plan Rejected"));
      
      if (!isDuplicate) {
        return [...prevMessages, { role: 'bot', content: resultContent }];
      }
      return prevMessages;
    });
  };

  // Reset form and start a new travel plan
  const startNewPlan = () => {
    setFormSubmitted(false);
    setInstanceId(null);
    setStatusPolling(false);
    setPlanReadyForApproval(false);
    setPlanData(null);
    setApprovalStatus(null);
    setMessages([]);
    setTravelRequest({
      userName: '',
      preferences: '',
      durationInDays: 7,
      budget: '',
      travelDates: '',
      specialRequirements: ''
    });
  };

  // Approve the travel plan
  const approveTravelPlan = async () => {
    if (!instanceId) return;
    
    setLoading(true);
    setApprovalStatus("processing");
    setPlanReadyForApproval(false);
    
    try {
      await axios.post(`${API_URL}/travel-planner/approve/${instanceId}`, {
        approved: true,
        comments: "The plan looks great! Looking forward to the trip."
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      // Add user approval message to the chat
      setMessages(prevMessages => [...prevMessages, { 
        role: 'user', 
        content: '✅ **I have approved the travel plan!** Proceeding with booking...'
      }]);
      
      // Continue polling to catch the booking status
      if (!statusPolling) {
        setStatusPolling(true);
      }
      
    } catch (error) {
      console.error('Error approving travel plan:', error);
      setApprovalStatus("waiting");
      setPlanReadyForApproval(true);
      setMessages(prevMessages => [...prevMessages, { 
        role: 'bot', 
        content: '❌ Error approving your travel plan. Please try again.'
      }]);
      setLoading(false);
    }
  };
  
  // Reject the travel plan
  const rejectTravelPlan = async () => {
    if (!instanceId) return;
    
    setLoading(true);
    setApprovalStatus("processing");
    setPlanReadyForApproval(false);
    
    try {
      await axios.post(`${API_URL}/travel-planner/approve/${instanceId}`, {
        approved: false,
        comments: "I'd like to consider other options or make changes to this plan."
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      // Add user rejection message to the chat
      setMessages(prevMessages => [...prevMessages, { 
        role: 'user', 
        content: '❌ **I have rejected the travel plan** and would like to make some changes.'
      }]);
      
      // Continue polling to catch the rejection response
      if (!statusPolling) {
        setStatusPolling(true);
      }
      
    } catch (error) {
      console.error('Error rejecting travel plan:', error);
      setApprovalStatus("waiting");
      setPlanReadyForApproval(true);
      setMessages(prevMessages => [...prevMessages, { 
        role: 'bot', 
        content: '❌ Error rejecting your travel plan. Please try again.'
      }]);
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="chat-title-container">
        <h1>Welcome to the Travel Planner Assistant</h1>
        {formSubmitted && <button onClick={startNewPlan} className="new-plan-btn">Start New Plan</button>}
      </div>
      
      {!formSubmitted ? (
        <div className="travel-form-container">
          <h2>Create Your Travel Plan</h2>
          
          <div className="form-group">
            <label>Name</label>
            <input
              type="text"
              name="userName"
              value={travelRequest.userName}
              onChange={handleInputChange}
              placeholder="e.g., Nick Greenfield"
            />
          </div>
          
          <div className="form-group">
            <label>Travel Preferences</label>
            <textarea
              name="preferences"
              value={travelRequest.preferences}
              onChange={handleInputChange}
              placeholder="e.g., Looking for a family-friendly luxury vacation with activities for children..."
              rows={4}
            />
          </div>
          
          <div className="form-group">
            <label>Duration (days)</label>
            <input
              type="number"
              name="durationInDays"
              value={travelRequest.durationInDays}
              onChange={handleInputChange}
              min="1"
              max="30"
            />
          </div>
          
          <div className="form-group">
            <label>Budget</label>
            <input
              type="text"
              name="budget"
              value={travelRequest.budget}
              onChange={handleInputChange}
              placeholder="e.g., Luxury, around $10000 total"
            />
          </div>
          
          <div className="form-group">
            <label>Travel Dates</label>
            <input
              type="text"
              name="travelDates"
              value={travelRequest.travelDates}
              onChange={handleInputChange}
              placeholder="e.g., July 1-11, 2025"
            />
          </div>
          
          <div className="form-group">
            <label>Special Requirements</label>
            <textarea
              name="specialRequirements"
              value={travelRequest.specialRequirements}
              onChange={handleInputChange}
              placeholder="e.g., Need connecting rooms or a family suite. Child has peanut allergy."
              rows={2}
            />
          </div>
          
          <button 
            onClick={submitTravelRequest} 
            className="submit-btn"
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Plan My Trip'}
          </button>
        </div>
      ) : (
        <div className="chat-container">
          <div ref={chatHistoryRef} className="chat-history">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            ))}
            
            {/* Show progress tracker only during initial planning phase and hide once plan is ready for approval or any later stages */}
            {statusPolling && !planReadyForApproval && !approvalStatus && (
              <div className="loading-container">
                {orchestrationStatus ? (
                  <ProgressTracker status={orchestrationStatus} />
                ) : (
                  <div className="loading-message">
                    Creating your personalized travel plan...
                    <br />
                    This may take a minute or two.
                  </div>
                )}
              </div>
            )}
          </div>
          
          {instanceId && planReadyForApproval && approvalStatus === "waiting" && confirmationStatus !== "confirmed" && (
            <div className="approve-section">
              <h3>Do you approve this travel plan?</h3>
              <p>If you approve, we'll proceed with booking your trip based on this plan.</p>
              <div className="approval-buttons">
                <button 
                  onClick={approveTravelPlan} 
                  className="approve-btn"
                  disabled={loading || approvalStatus === "processing"}
                >
                  Yes, Book My Trip!
                </button>
                <button 
                  onClick={rejectTravelPlan} 
                  className="reject-btn"
                  disabled={loading || approvalStatus === "processing"}
                >
                  No, I Need Changes
                </button>
              </div>
            </div>
          )}
          
          {approvalStatus === "rejected" && (
            <div className="approve-section">
              <button onClick={startNewPlan} className="new-plan-btn full-width">
                Start a New Travel Plan
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ChatInterface;