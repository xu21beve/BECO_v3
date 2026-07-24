#
#	Author:			Sebastien Parent-Charette (support@robotshop.com)
#	Version:		1.0.0
#	Licence:		LGPL-3.0 (GNU Lesser General Public License version 3)
#	
#	Desscription:	Basic example of the LSS moving back and forth.
#

# Import required liraries
import time
import serial

# Import LSS library
import lib.lss.lss as lss
import lib.lss.lss_const as lssc

# Constants
#CST_LSS_Port = "/dev/ttyUSB0"		# For Linux/Unix platforms
CST_LSS_Port = "COM10"				# For windows platforms
CST_LSS_Baud = lssc.LSS_DefaultBaud

# Create and open a serial port
lss.initBus(CST_LSS_Port, CST_LSS_Baud)

# Create an LSS object
myLSS = lss.LSS(0)
myLSS1 = lss.LSS(1)
myLSS2 = lss.LSS(2)

# Initialize LSS to position 0.0 deg
myLSS2.move(0)

# Wait for it to get there
time.sleep(2)

# Loops between -180.0 deg and 180 deg, taking 1 second pause between each half-circle move.
while 1:
	# Send LSS to half a turn counter-clockwise from zero (assumes gyre = 1)
	myLSS2.move(-600)
	
	# Wait for two seconds
	time.sleep(2)
	
	# Send LSS to half a turn clockwise from zero (assumes gyre = 1)
	myLSS2.move(600)
	
	# Wait for two seconds
	time.sleep(2)

### EOF #######################################################################